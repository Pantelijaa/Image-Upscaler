import argparse
import glob
import os
import re

import numpy as np
import onnx
import onnxruntime as ort
import torch

from models import load_checkpoint


def verify(onnx_path):
	model = onnx.load(onnx_path)
	onnx.checker.check_model(model)
	print(f"ONNX model valid: {onnx_path}")
	print(f"  Opset version: {model.opset_import[0].version}")
	print(f"  IR version: {model.ir_version}")

	for node in model.graph.node:
		print(f"  Node: {node.op_type}")

	session = ort.InferenceSession(onnx_path)
	dummy = np.random.randn(1, 1, 64, 64).astype(np.float32)
	output = session.run(None, {"input": dummy})
	print(f"  Input shape:  {dummy.shape}")
	print(f"  Output shape: {output[0].shape}")
	print("  Inference OK")


def get_next_version(onnx_dir, arch):
	existing = glob.glob(os.path.join(onnx_dir, f"{arch}_v*.onnx"))
	versions = []
	for path in existing:
		match = re.search(rf"{re.escape(arch)}_v(\d+)\.onnx$", os.path.basename(path))
		if match:
			versions.append(int(match.group(1)))
	return max(versions, default=0) + 1


def export_onnx(weights_path, output_dir="models/onnx", patch_size=33):
	os.makedirs(output_dir, exist_ok=True)

	model, ckpt = load_checkpoint(weights_path)
	model.eval()
	arch = ckpt["arch"]

	version = get_next_version(output_dir, arch)
	onnx_path = os.path.join(output_dir, f"{arch}_v{version}.onnx")

	dummy_input = torch.randn(1, 1, patch_size, patch_size)

	torch.onnx.export(
		model,
		dummy_input,
		onnx_path,
		export_params=True,
		opset_version=11,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["output"],
		dynamic_axes={
			"input": {2: "height", 3: "width"},
			"output": {2: "height", 3: "width"},
		},
		dynamo=False,
	)

	print(f"Exported {arch} (scale {ckpt.get('scale', '?')}x) as v{version}: {onnx_path}", flush=True)
	return onnx_path, version


def parse_args():
	parser = argparse.ArgumentParser(description="Export trained checkpoints to ONNX.")
	parser.add_argument("checkpoints", nargs="*",
	                    help="checkpoint files to export (default: every models/pth/best_*.pth)")
	parser.add_argument("--out", default="models/onnx", help="output directory")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	paths = args.checkpoints or sorted(glob.glob(os.path.join("models", "pth", "best_*.pth")))
	if not paths:
		raise SystemExit("No checkpoints found in models/pth. Train a model first.")
	for path in paths:
		onnx_path, _ = export_onnx(path, args.out)
		verify(onnx_path)
