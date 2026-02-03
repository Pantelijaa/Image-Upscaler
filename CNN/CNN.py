import torch
import torch.nn as nn

class CNN(nn.Module):
	def __init__(self):
		supet(CNN, self).__init__()

		self.conv1 = nn.Conv2d(1, 64, 9, 4)
		self.conv2 = nn.Conv2d(64, 32, 5, 2)
		self.conv3 = nn.Conv2d(32, 1, 5, 2)
		self.relu = nn.ReLU(inplace=True)

		def forward(self, x):
			x = self.relu(self.conv1(x))
			x = self.relu(self.conv2(x))
			x = self.conv3(x)
			return x