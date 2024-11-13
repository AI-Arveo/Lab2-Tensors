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


class NeuralNetworkClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """ START TODO: fill in all three layers. Remember that each layer should contain 2 parts, a linear transformation and a nonlinear activation function."""
        self.layer1 = nn.Sequential(
            #nn.Linear(in_features=1, out_features=1),  # Klopt dit? Waar vind ik dit?
            nn.Linear(in_features=784, out_features=32), # activation function
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU() #alpha => Default: 1.0
        )

        self.layer3 = nn.Sequential(
            nn.Linear(in_features=16, out_features=10),  # assuming 3 classes
            nn.Softmax(dim=1)  # Softmax along the class dimension -> dimension along which Softmax will be computed(so every slice along dim will sum to 1).
        )
        """END TODO"""

    def forward(self, x: torch.Tensor):
        """START TODO: forward tensor x through all layers."""
        x = self.layer1(x.flatten(1))
        x = self.layer2(x)
        x = self.layer3(x)
        """END TODO"""
        return x


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
