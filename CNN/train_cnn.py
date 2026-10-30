import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from SRDataset import SRDataset
from models import MODELS, build_model, default_lr, save_checkpoint
from export_onnx import export_onnx


def train(data_dir, arch="vdsr", train_minutes=240, batch_size=64, lr=None, val_split=0.2, scale=2):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if lr is None:
		lr = default_lr(arch)

	print("\nLoading dataset...", flush=True)
	dataset = SRDataset(data_dir, patch_size=33, scale=scale, stride=14)
	val_size = int(len(dataset) * val_split)
	train_size = len(dataset) - val_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	model, config = build_model(arch)
	model = model.to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.param_groups(lr))

	# Reduce LR when validation loss plateaus
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", factor=0.5, patience=10
	)

	os.makedirs("models/pth", exist_ok=True)
	checkpoint_path = f"models/pth/best_{arch}.pth"

	best_val_loss = float("inf")
	epoch = 0
	time_limit = train_minutes * 60
	num_params = sum(p.numel() for p in model.parameters())

	print(f"\n    Device:             {device}")
	print(f"    Architecture:       {arch} {config} ({num_params:,} params)")
	print(f"    Scale factor:       {scale}x")
	print(f"    Batch size:         {batch_size}")
	print(f"    Train patches:      {train_size}")
	print(f"    Validation patches: {val_size}")
	print(f"    Learning rate:      {lr}")
	print(f"    Train duration:     {train_minutes} min", flush=True)

	start_time = time.time()

	while True:
		elapsed = time.time() - start_time
		if elapsed >= time_limit:
			print("Time limit reached. Stopping training.")
			break
		epoch += 1
		model.train()
		train_loss = 0.0
		for lr_patches, hr_patches in train_loader:
			lr_patches = lr_patches.to(device, non_blocking=True)
			hr_patches = hr_patches.to(device, non_blocking=True)

			predictions = model(lr_patches)
			loss = criterion(predictions, hr_patches)

			optimizer.zero_grad()
			loss.backward()

			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

			optimizer.step()

			train_loss += loss.item() * lr_patches.size(0)

			if time.time() - start_time >= time_limit:
				break

		train_loss /= train_size

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for lr_patches, hr_patches in val_loader:
				lr_patches = lr_patches.to(device, non_blocking=True)
				hr_patches = hr_patches.to(device, non_blocking=True)

				predictions = model(lr_patches)
				loss = criterion(predictions, hr_patches)
				val_loss += loss.item() * lr_patches.size(0)
		val_loss /= val_size

		# Step scheduler based on validation loss
		scheduler.step(val_loss)

		elapsed = time.time() - start_time
		remaining = max(0, time_limit - elapsed)
		current_lr = optimizer.param_groups[0]["lr"]
		print(
			f"Epoch {epoch}  ",
			f"train_loss: {train_loss:.6f}  ",
			f"val_loss: {val_loss:.6f}  ",
			f"lr: {current_lr:.6f}  ",
			f"[{elapsed:.0f}s elapsed, {remaining:.0f}s remaining]",
			flush=True
		)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			save_checkpoint(
				checkpoint_path, model, arch, config, scale,
				extra={"epoch": epoch, "val_loss": val_loss},
			)
			print(f"  -> saved best model (val_loss={best_val_loss:.6f})")
	total = time.time() - start_time
	print(f"Training complete. {epoch} epochs in {total:.1f}s. Best val_loss: {best_val_loss:.6f}")

	onnx_path, version = export_onnx(checkpoint_path, "models/onnx")
	print(f"Ready for C++ inference: {onnx_path}", flush=True)


def parse_args():
	parser = argparse.ArgumentParser(description="Train a super-resolution network.")
	parser.add_argument("--arch", choices=list(MODELS), default="vdsr",
	                    help="network architecture (default: vdsr)")
	parser.add_argument("--data", default=os.path.join("data", "train", "HR"),
	                    help="directory with high-resolution training images")
	parser.add_argument("--minutes", type=float, default=240, help="training time budget")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=None,
	                    help="learning rate (default: per-architecture value from models.py)")
	parser.add_argument("--scale", type=int, default=2, help="upscaling factor")
	parser.add_argument("--val-split", type=float, default=0.2)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	train(
		args.data,
		arch=args.arch,
		train_minutes=args.minutes,
		batch_size=args.batch_size,
		lr=args.lr,
		val_split=args.val_split,
		scale=args.scale,
	)
