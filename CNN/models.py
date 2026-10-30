"""
Super-resolution network architectures.

Both models take a single-channel (luminance) image that has already been
upscaled to the target size with bicubic interpolation, and return an image
of the same size. Shape: (N, 1, H, W) -> (N, 1, H, W).

	srcnn  - the original three-layer SRCNN (Dong et al., 2014), 9-5-5 kernels,
	         64 and 32 feature maps, no residual connection.
	vdsr   - a VDSR-style deep residual network (Kim et al., 2016) with 3x3
	         kernels. The default depth is 8 layers, shallower than the
	         20-layer original, hence "VDSR-lite" in the README.

Use build_model(name) to construct one by name; MODELS lists the names.
"""

import torch
import torch.nn as nn


class SRCNN(nn.Module):
	"""
	SRCNN 9-5-5 (Dong et al., "Image Super-Resolution Using Deep Convolutional
	Networks", 2014).

	  patch extraction : conv 9x9,  1 -> n1 (64)
	  non-linear map   : conv 5x5, n1 -> n2 (32)
	  reconstruction   : conv 5x5, n2 -> 1

	The paper uses no padding (output smaller than input). Padding is added
	here so the output has the same size as the input, which the C++
	inference side relies on.
	"""

	name = "srcnn"

	def __init__(self, n1=64, n2=32, k1=9, k2=5, k3=5):
		super().__init__()
		self.patch_extraction = nn.Conv2d(1, n1, kernel_size=k1, padding=k1 // 2)
		self.non_linear_map = nn.Conv2d(n1, n2, kernel_size=k2, padding=k2 // 2)
		self.reconstruction = nn.Conv2d(n2, 1, kernel_size=k3, padding=k3 // 2)
		self.relu = nn.ReLU(inplace=True)
		self._initialize_weights()

	def _initialize_weights(self):
		# Paper: Gaussian, std 0.001, zero bias.
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight, mean=0.0, std=0.001)
				nn.init.zeros_(m.bias)

	def param_groups(self, lr):
		"""
		Paper trains the reconstruction layer with a 10x smaller learning rate.
		"""
		return [
			{"params": self.patch_extraction.parameters(), "lr": lr},
			{"params": self.non_linear_map.parameters(), "lr": lr},
			{"params": self.reconstruction.parameters(), "lr": lr * 0.1},
		]

	def forward(self, x):
		out = self.relu(self.patch_extraction(x))
		out = self.relu(self.non_linear_map(out))
		return self.reconstruction(out)


class VDSR(nn.Module):
	"""
	VDSR-style residual network (Kim et al., "Accurate Image Super-Resolution
	Using Very Deep Convolutional Networks", 2016).

	  num_layers conv 3x3 layers with num_features channels and ReLU between
	  them; the network predicts the residual that is added to the bicubic
	  input. The original uses 20 layers; the default here is 8.
	"""

	name = "vdsr"

	def __init__(self, num_layers=8, num_features=64):
		super().__init__()
		if num_layers < 2:
			raise ValueError("VDSR needs at least 2 layers")

		layers = [nn.Conv2d(1, num_features, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
		for _ in range(num_layers - 2):
			layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))

		self.network = nn.Sequential(*layers)
		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
				nn.init.zeros_(m.bias)

	def param_groups(self, lr):
		return [{"params": self.parameters(), "lr": lr}]

	def forward(self, x):
		return self.network(x) + x


# name -> (class, default constructor kwargs, default learning rate)
MODELS = {
	"srcnn": (SRCNN, {}, 1e-4),
	"vdsr": (VDSR, {"num_layers": 8, "num_features": 64}, 1e-3),
}


def build_model(name, **overrides):
	"""Construct a model by name. Keyword overrides go to the constructor."""
	if name not in MODELS:
		raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODELS)}")
	cls, defaults, _ = MODELS[name]
	kwargs = {**defaults, **overrides}
	return cls(**kwargs), kwargs


def default_lr(name):
	return MODELS[name][2]


def save_checkpoint(path, model, arch, config, scale, extra=None):
	"""Save weights together with what is needed to rebuild the model."""
	payload = {
		"arch": arch,
		"config": config,
		"scale": scale,
		"state_dict": model.state_dict(),
	}
	if extra:
		payload.update(extra)
	torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
	"""Rebuild a model from a checkpoint written by save_checkpoint."""
	ckpt = torch.load(path, map_location=map_location, weights_only=True)
	model, config = build_model(ckpt["arch"], **ckpt["config"])
	model.load_state_dict(ckpt["state_dict"])
	return model, ckpt


if __name__ == "__main__":
	for name in MODELS:
		model, cfg = build_model(name)
		params = sum(p.numel() for p in model.parameters())
		x = torch.rand(1, 1, 33, 33)
		y = model(x)
		print(f"{name:6s} params={params:>8,d}  config={cfg}  in={tuple(x.shape)} out={tuple(y.shape)}")
