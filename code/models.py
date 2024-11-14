import torch
from torch import nn


class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """START TODO: define a linear layer"""
        # in_features is 1 omdat je bij datasets ziet dat de huis_prijs enkel afhankelijk is van de huis_size
        self.linear_layer = nn.Linear(in_features=1, out_features=1)
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward the tensor x through the linear layer and return the output"""
        return self.linear_layer(x)
        """END TODO"""


class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # First conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling by 2x
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # Second conv layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # Forward pass through convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)

        # Flatten before passing through fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return nn.Softmax(dim=1)(x)


class NeuralNetworkClassificationModelWithVariableLayers(torch.nn.Module):
    def __init__(self, in_size, out_size, hidden_sizes = []):
        super().__init__()

        self.layers = torch.nn.Sequential()

        for i, size in enumerate(hidden_sizes):
            self.layers.add_module(f"lin_layer_{i + 1}", torch.nn.Linear(in_size, size))
            self.layers.add_module(f"relu_layer_{i + 1}", torch.nn.ReLU())
            in_size = size

        self.layers.add_module(f"lin_layer_{len(hidden_sizes) + 1}", torch.nn.Linear(in_size, out_size))
        self.layers.add_module(f"softmax_layer", torch.nn.Softmax(dim=1))

    def forward(self, x: torch.Tensor):
        x = x.flatten(1)
        x = self.layers(x)
        return x
