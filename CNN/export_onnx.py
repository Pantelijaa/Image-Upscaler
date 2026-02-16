import torch
import os
import glob
import re
from SRCNN import SRCNN

def export_onnx(weights_path="models/pth/best_srcnn.pth", outpur_dir="models/onnx", patch_size=33):
	os.makedirs(outpur_dir, exist_ok=True)
	
	model = SRCNN()
	print(weights_path)
	model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
	model.eval()

	onnx_path = os.path.join(outpur_dir, "srcnn.onnx")

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

	print(f"Exported ONNX model: {onnx_path}", flush=True)
	return onnx_path

if __name__ == "__main__":
	os.environ["PYTHONIOENCODING"] = "utf-8"
	export_onnx()