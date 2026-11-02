import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from SRDataset import SRDataset
from models import MODELS, build_model, default_lr, save_checkpoint
from export_onnx import export_onnx


def train(data_dir, arch="vdsr", train_minutes=60, batch_size=64, lr=None, val_split=0.2, scale=2, stride=14, seed=0, resume=False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if lr is None:
		lr = default_lr(arch)
	torch.manual_seed(seed)

	os.makedirs("models/pth", exist_ok=True)
	best_path = f"models/pth/best_{arch}_x{scale}.pth"
	last_path = f"models/pth/last_{arch}_x{scale}.pth"

	# Resume continues from the last epoch's full state (weights, optimizer, scheduler, epoch counter, best loss); the time budget applies to this session.
	ckpt = None
	if resume:
		if not os.path.exists(last_path):
			raise SystemExit(f"Nothing to resume: {last_path} not found.")
		ckpt = torch.load(last_path, map_location=device, weights_only=True)
		if ckpt["seed"] != seed:
			raise SystemExit(f"{last_path} was trained with --seed {ckpt['seed']}; pass the same seed so the train/val split matches.")

	print("\nLoading dataset...", flush=True)
	dataset = SRDataset(data_dir, patch_size=33, scale=scale, stride=stride)
	val_size = int(len(dataset) * val_split)
	train_size = len(dataset) - val_size
	split = torch.Generator().manual_seed(seed)
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split)

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	model, config = build_model(arch, **(ckpt["config"] if ckpt else {}))
	model = model.to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.param_groups(lr))

	# Reduce LR when validation loss plateaus
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer, mode="min", factor=0.5, patience=10
	)

	best_val_loss = float("inf")
	epoch = 0
	if ckpt:
		model.load_state_dict(ckpt["state_dict"])
		optimizer.load_state_dict(ckpt["optimizer"])
		scheduler.load_state_dict(ckpt["scheduler"])
		epoch = ckpt["epoch"]
		best_val_loss = ckpt["best_val_loss"]
		print(f"Resumed from {last_path}: {epoch} epochs done, best val_loss {best_val_loss:.6f}")

	time_limit = train_minutes * 60
	num_params = sum(p.numel() for p in model.parameters())

	print(f"\n    Device:             {device}")
	print(f"    Architecture:       {arch} {config} ({num_params:,} params)")
	print(f"    Scale factor:       {scale}x")
	print(f"    Batch size:         {batch_size}")
	print(f"    Train patches:      {train_size}")
	print(f"    Validation patches: {val_size}")
	print(f"    Learning rate:      {optimizer.param_groups[0]['lr']}")
	print(f"    Seed:               {seed}")
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

		improved = val_loss < best_val_loss
		best_val_loss = min(best_val_loss, val_loss)
		state = {
			"epoch": epoch,
			"val_loss": val_loss,
			"best_val_loss": best_val_loss,
			"seed": seed,
			"optimizer": optimizer.state_dict(),
			"scheduler": scheduler.state_dict(),
		}
		save_checkpoint(last_path, model, arch, config, scale, extra=state)
		if improved:
			save_checkpoint(best_path, model, arch, config, scale, extra={"epoch": epoch, "val_loss": val_loss, "seed": seed})
			print(f"  -> saved best model (val_loss={best_val_loss:.6f})")
	total = time.time() - start_time
	print(f"Training complete. {epoch} epochs in {total:.1f}s. Best val_loss: {best_val_loss:.6f}")

	onnx_path, version = export_onnx(best_path, "models/onnx")
	print(f"Ready for C++ inference: {onnx_path}", flush=True)


def parse_args():
	parser = argparse.ArgumentParser(description="Train a super-resolution network.")
	parser.add_argument("--arch", choices=list(MODELS), default="vdsr",
	                    help="network architecture (default: vdsr)")
	parser.add_argument("--data", default=os.path.join("data", "train", "HR"),
	                    help="directory with high-resolution training images")
	parser.add_argument("--minutes", type=float, default=60, help="training time budget for this session")
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=None,
	                    help="learning rate (default: per-architecture value from models.py; ignored with --resume)")
	parser.add_argument("--scale", type=int, default=2, help="upscaling factor")
	parser.add_argument("--val-split", type=float, default=0.2)
	parser.add_argument("--stride", type=int, default=14,
	                    help="patch extraction stride; larger = fewer patches, less memory")
	parser.add_argument("--seed", type=int, default=0, help="seed for weight init and the train/val split")
	parser.add_argument("--resume", action="store_true",
	                    help="continue from models/pth/last_<arch>_x<scale>.pth instead of starting over")
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
		stride=args.stride,
		seed=args.seed,
		resume=args.resume,
	)
