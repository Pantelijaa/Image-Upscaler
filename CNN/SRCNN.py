import torch
import torch.nn as nn

class SRCNN(nn.Module):
	def __init__(self, num_layers=8, num_features=64):
		super(SRCNN, self).__init__()

		layers = []

		#First layer: 1-> num_features
		layers.append(nn.Conv2d(1, num_features, kernel_size=3, padding=1))
		layers.append(nn.ReLU(inplace=False))

		# Hidden layers: num_features -> num_features
		for _ in range(num_layers - 2):
			layers.append(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1))
			layers.append(nn.ReLU(inplace=False))

		# Final layer: num_features -> 1
		layers.append(nn.Conv2d(num_features, 1, kernel_size=3, padding=1))

		self.network = nn.Sequential(*layers)

		self._initialize_weights()

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
				if m.bias is not None:
					nn.init.zeros_(m.bias)


	def forward(self, x):
		residual = x
		out = self.network(x)
		return out + residual
