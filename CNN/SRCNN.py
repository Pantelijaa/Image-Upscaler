import torch
import torch.nn as nn

class SRCNN(nn.Module):
	def __init__(self):
		super(SRCNN, self).__init__()

		self.conv1 = nn.Conv2d(1, 128, kernel_size=9, padding=4)
		self.conv2 = nn.Conv2d(128, 64, kernel_size=5, padding=2)
		self.conv3 = nn.Conv2d(64, 1, kernel_size=5, padding=2)
		self.relu = nn.ReLU(inplace=True)

	def forward(self, x):
		residual = x
		out = self.relu(self.conv1(x))
		out = self.relu(self.conv2(out))
		out = self.conv3(out)
		return out + residual
