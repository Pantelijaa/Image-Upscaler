import torch
import os
import glob
import re
from SRCNN import SRCNN
import onnx
import onnxruntime as ort
import numpy as np

def verify(onnx_path="models/onnx/srcnn_v1.onnx"):
    # Step 1: Check ONNX structure
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print(f"ONNX model valid: {onnx_path}")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  IR version: {model.ir_version}")

    # Step 2: Print graph nodes (to check for unsupported ops)
    for node in model.graph.node:
        print(f"  Node: {node.op_type}")

    # Step 3: Run inference to confirm it works
    session = ort.InferenceSession(onnx_path)
    dummy = np.random.randn(1, 1, 64, 64).astype(np.float32)
    output = session.run(None, {"input": dummy})
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {output[0].shape}")
    print("  Inference OK")

def get_next_vestion(onnx_dir):
	existing = glob.glob(os.path.join(onnx_dir, "srcnn_v*.onnx"))
	versions = []
	for path in existing:
		match = re.search(r"srcnn_v(\d+)\.onnx", os.path.basename(path))
		if match:
			versions.append(int(match.group(1)))
	return max(versions, default=0) + 1

def export_onnx(weights_path="models/pth/best_srcnn.pth", outpur_dir="models/onnx", patch_size=33):
	os.makedirs(outpur_dir, exist_ok=True)
	
	model = SRCNN()
	print(weights_path)
	model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
	model.eval()

	version = get_next_vestion(outpur_dir)
	onnx_path = os.path.join(outpur_dir, f"srcnn_v{version}.onnx")

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
            "input":  {2: "height", 3: "width"},
            "output": {2: "height", 3: "width"},
        },
	)

	print(f"Exported ONNX model v{version}: {onnx_path}", flush=True)
	return onnx_path, version

if __name__ == "__main__":
	export_onnx()
	verify()
