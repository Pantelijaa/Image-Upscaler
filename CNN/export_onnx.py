import argparse
import glob
import os
import re

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from PIL import Image

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


def get_next_version(onnx_dir, stem):
	existing = glob.glob(os.path.join(onnx_dir, f"{stem}_v*.onnx"))
	versions = []
	for path in existing:
		match = re.search(rf"{re.escape(stem)}_v(\d+)\.onnx$", os.path.basename(path))
		if match:
			versions.append(int(match.group(1)))
	return max(versions, default=0) + 1


def scale_from_name(onnx_path):
	match = re.search(r"_x(\d+)(_|\.)", os.path.basename(onnx_path))
	if not match:
		raise ValueError(f"Cannot read the scale factor from the file name: {onnx_path}")
	return int(match.group(1))


def export_onnx(weights_path, output_dir="models/onnx", patch_size=33):
	os.makedirs(output_dir, exist_ok=True)

	model, ckpt = load_checkpoint(weights_path)
	model.eval()
	arch = ckpt["arch"]

	stem = f"{arch}_x{ckpt['scale']}"  # <arch>_x<scale>_vN.onnx: the benchmark runs the model at this scale only
	version = get_next_version(output_dir, stem)
	onnx_path = os.path.join(output_dir, f"{stem}_v{version}.onnx")

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


class CalibrationReader(CalibrationDataReader):
	"""
	Network inputs for calibration: training images degraded exactly like the
	benchmark (bicubic down by `scale`, bicubic up, luminance in [0, 1]).
	"""

	def __init__(self, image_dir, scale, num_images):
		files = sorted(glob.glob(os.path.join(image_dir, "*.png")))
		if not files:
			raise FileNotFoundError(f"No calibration images in {image_dir}. Build the prepare_data target first.")
		step = max(1, len(files) // num_images)
		self.files = files[::step][:num_images]
		self.scale = scale
		self.it = None

	def _input(self, path):
		img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
		h, w = img.shape
		h -= h % self.scale
		w -= w % self.scale
		img = img[:h, :w]
		lr = np.asarray(Image.fromarray(img).resize((w // self.scale, h // self.scale), Image.BICUBIC))
		up = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
		return {"input": (up.astype(np.float32) / 255.0)[None, None]}

	def get_next(self):
		if self.it is None:
			self.it = (self._input(p) for p in self.files)
		return next(self.it, None)


def quantize_int8(onnx_path, calib_dir=os.path.join("data", "train", "HR"), num_images=20):
	out_path = re.sub(r"\.onnx$", "_int8.onnx", onnx_path)
	reader = CalibrationReader(calib_dir, scale_from_name(onnx_path), num_images)
	quantize_static(
		onnx_path,
		out_path,
		reader,
		quant_format=QuantFormat.QOperator,
		activation_type=QuantType.QUInt8,
		weight_type=QuantType.QInt8,
		per_channel=True,
	)
	print(f"Quantized to int8 ({len(reader.files)} calibration images): {out_path}", flush=True)
	return out_path


def parse_args():
	parser = argparse.ArgumentParser(description="Export trained checkpoints to ONNX, optionally quantized to int8.")
	parser.add_argument("paths", nargs="*",
	                    help=".pth checkpoints to export and/or float .onnx models to quantize "
	                         "(default: every models/pth/best_*.pth, or with --int8 alone every un-quantized models/onnx/*.onnx)")
	parser.add_argument("--out", default="models/onnx", help="output directory")
	parser.add_argument("--int8", action="store_true", help="also write an int8-quantized copy (<name>_int8.onnx)")
	parser.add_argument("--calib-images", type=int, default=20, help="training images used for calibration")
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	paths = args.paths
	if not paths:
		paths = sorted(glob.glob(os.path.join("models", "pth", "best_*.pth")))
		if not paths and args.int8:
			paths = [p for p in sorted(glob.glob(os.path.join(args.out, "*.onnx"))) if not p.endswith("_int8.onnx")]
	if not paths:
		raise SystemExit("Nothing to do: no checkpoints in models/pth" + (" and no float models in models/onnx." if args.int8 else "."))

	for path in paths:
		if path.endswith(".onnx"):
			if not args.int8:
				raise SystemExit(f"{path} is already ONNX; pass --int8 to quantize it.")
			verify(quantize_int8(path, num_images=args.calib_images))
			continue
		onnx_path, _ = export_onnx(path, args.out)
		verify(onnx_path)
		if args.int8:
			verify(quantize_int8(onnx_path, num_images=args.calib_images))
