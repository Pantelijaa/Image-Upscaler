from SRDataset import SRDataset
from SRCNN import SRCNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import time
from export_onnx import export_onnx

def train(data_dir, train_minutes = 3, batch_size=128, lr=1e-4, val_split=0.2, scale=2):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Loading dataset...", flush=True)
	dataset = SRDataset(data_dir, patch_size=33, scale=scale, stride=14)
	val_size = int(len(dataset) * val_split)
	train_size = len(dataset) - val_size
	train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

	
	model = SRCNN().to(device)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	best_val_loss = float("inf")
	epoch = 0
	time_limit = train_minutes * 60

	print(f"\n\tDevice:         {device}")
	print(f"    Scale factor:   {scale}x")
	print(f"    Batch size:     {batch_size}")
	print(f"    Train patches:  {train_size}")
	print(f"    Validation patches:	{val_size}")
	print(f"    Learning rate:  {lr}")
	print(f"    Train duration: {train_minutes} min", flush=True)

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
			lr_patches = lr_patches.to(device)
			hr_patches = hr_patches.to(device)

			predictions = model(lr_patches)
			loss = criterion(predictions, hr_patches)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			train_loss += loss.item() * lr_patches.size(0)

			if time.time() - start_time >= time_limit:
				break

		train_loss /= train_size

		model.eval()
		val_loss = 0.0
		with torch.no_grad():
			for lr_patches, hr_patches in val_loader:
				lr_patches = lr_patches.to(device)
				hr_patches = hr_patches.to(device)

				predictions = model(lr_patches)
				loss = criterion(predictions, hr_patches)
				val_loss += loss.item() * lr_patches.size(0)
		val_loss /= val_size

		elapsed = time.time() - start_time
		remaining = max(0, time_limit - elapsed)
		print(
			f"Epoch {epoch}  ",
			f"train_loss: {train_loss:.6f}  ",
			f"val_loss: {val_loss:.6f}  ",
			f"[{elapsed:.0f}s elapsed, {remaining:.0f}s remaining]",
			flush = True
		)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(model.state_dict(), "models/pth/best_srcnn.pth")
			print(f"  -> saved best model (val_loss={best_val_loss:.6f})")
	total = time.time() - start_time
	print(f"Training complete. {epoch} epochs in {total:.1f}s. Best val_loss: {best_val_loss:.6f}")

	onnx_path, version = export_onnx("models/pth/best_srcnn.pth", "models/onnx")
	print(f"Ready for C++ inference: {onnx_path}", flush=True)


if __name__ == "__main__":
	train("data\\train\\HR")