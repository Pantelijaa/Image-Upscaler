import os
import cv2
import numpy as np
import glob
import torch
from PIL import Image
from torch.utils.data import Dataset


class SRDataset(Dataset):
	"""
	Patch-based super-resolution dataset.

	Every image is loaded as luminance (grayscale), augmented 8x (4 rotations x
	flip) and cut into overlapping patch_size x patch_size patches. For each HR
	patch the LR patch is the same region after the benchmark degradation:
	bicubic downscale by `scale` (Pillow, antialiased, as in prepare_data.py)
	followed by bicubic upscale back to the original size (OpenCV INTER_CUBIC,
	as in the C++ inference path).

	Patches are kept as uint8 in two big tensors and converted to float32 in
	[0, 1] on access, so the 291-image T91+BSD200 set fits in a few GB.
	"""

	def __init__(self, image_dir, patch_size=33, scale=2, stride=14):
		super().__init__()
		self.scale = scale
		self.patch_size = patch_size

		files = sorted(
			glob.glob(os.path.join(image_dir, "*.png"))
			+ glob.glob(os.path.join(image_dir, "*.jpg"))
			+ glob.glob(os.path.join(image_dir, "*.bmp"))
		)
		if not files:
			raise FileNotFoundError(f"No training images found in {image_dir}. Run CNN/prepare_data.py first.")

		lr_chunks, hr_chunks = [], []
		for path in files:
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue
			for augmented_image in self._augment(img):
				lr, hr = self._extract_patches(augmented_image, stride)
				if lr is not None:
					lr_chunks.append(lr)
					hr_chunks.append(hr)

		# (N, 1, patch, patch) uint8
		self.patches_lr = torch.from_numpy(np.concatenate(lr_chunks)).unsqueeze(1)
		self.patches_hr = torch.from_numpy(np.concatenate(hr_chunks)).unsqueeze(1)

	@staticmethod
	def _augment(img):
		"""
		8 augmented versions of an image
		(original + 3 rotations + horizontal flip + 3 flipped rotations)
		"""
		variants = []
		for rot in range(4):
			rotated = np.rot90(img, k=rot)
			variants.append(np.ascontiguousarray(rotated))
			variants.append(np.ascontiguousarray(np.fliplr(rotated)))
		return variants

	def _extract_patches(self, img, stride):
		h, w = img.shape
		w -= w % self.scale
		h -= h % self.scale
		img = img[:h, :w]

		if h < self.patch_size or w < self.patch_size:
			return None, None

		# Degradation must match the benchmark:
		#   1) downscale with antialiased bicubic (Pillow), the same as
		#      prepare_data.py uses to build the LR test images;
		#   2) upscale back with OpenCV INTER_CUBIC, the same as the C++
		#      SRCNNUpscaler does before running the network.
		lr_small = np.asarray(
			Image.fromarray(img).resize((w // self.scale, h // self.scale), Image.BICUBIC)
		)
		lr_up = cv2.resize(lr_small, (w, h), interpolation=cv2.INTER_CUBIC)

		# All patch windows at once via stride tricks: (rows, cols, p, p) -> (N, p, p)
		p = self.patch_size
		hr_patches = np.lib.stride_tricks.sliding_window_view(img, (p, p))[::stride, ::stride].reshape(-1, p, p)
		lr_patches = np.lib.stride_tricks.sliding_window_view(lr_up, (p, p))[::stride, ::stride].reshape(-1, p, p)
		return np.ascontiguousarray(lr_patches), np.ascontiguousarray(hr_patches)

	def __len__(self):
		return self.patches_lr.shape[0]

	def __getitem__(self, idx):
		lr = self.patches_lr[idx].float().div_(255.0)
		hr = self.patches_hr[idx].float().div_(255.0)
		return lr, hr
