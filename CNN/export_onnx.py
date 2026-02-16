import torch
import os
import glob
import re
from SRCNN import SRCNN

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
		opset_version=18,
		do_constant_folding=True,
		input_names=["input"],
		output_names=["output"],
		dynamic_axes={
			"input":  {0: "batch", 2: "height", 3: "width"},
			"output": {0: "batch", 2: "height", 3: "width"},
		},
	)

	print(f"Exported ONNX model v{version}: {onnx_path}", flush=True)
	return onnx_path, version

if __name__ == "__main__":
	export_onnx()