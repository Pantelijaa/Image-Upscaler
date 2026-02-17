import os
import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
	def __init__(self, image_dir, patch_size=33, scale=2, stride=14):
		super().__init__();
		self.scale = scale
		self.patch_size = patch_size
		self.patches_lr = []
		self.patches_hr = []

		files = sorted(
			glob.glob(os.path.join(image_dir, "*.png"))
			+ glob.glob(os.path.join(image_dir, "*.jpg"))
			+ glob.glob(os.path.join(image_dir, "*.bmp"))
		)

		for path in files:
			img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
			if img is None:
				continue

			augumented = self._augument(img)

			for augumented_image in augumented
				self._extract_patches(augumented_image, stride)

		
	def _augument(self, img):
		"""
		Generate 8  augumented versions of image 
		(original		+ 3 rotations + 
		horizontal flip + 3 flipped rotations)
		"""
		variants = []
		for rot in range(4):
			rotated = np.rot90(img, k=rot)
			variants.append(rotated)
			variants.append(np.fliplr(rotated))
		return variants

	def _extract_patches(self, img, stride):
		h, w = img.shape
		w -= w % self.scale
		h -= h % self.scale
		img = img[:h, :w]

		if h < self.patch_size or w < self.patch_size:
			return

		# normalize to [0, 1]
		hr = img.astype(np.float32) / 255.0

		lr = cv2.resize(hr, (w // self.scale, h // self.scale), interpolation=cv2.INTER_CUBIC)
		lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)

		# extract patches
		for y in range(0, h - self.patch_size + 1, stride):
			for x in range(0, w - self.patch_size + 1, stride):
				hr_patch = hr[y:y+ self.patch_size, x:x + self.patch_size]
				lr_patch = lr[y:y + self.patch_size, x:x + self.patch_size]

				# tensor shape: (1, H, W)
				self.patches_hr.append(torch.from_numpy(hr_patch).unsqueeze(0))
				self.patches_lr.append(torch.from_numpy(lr_patch).unsqueeze(0))
	
	def __len__(self):
		return len(self.patches_lr)

	def __getitem__(self, idx):
		return self.patches_lr[idx], self.patches_hr[idx]