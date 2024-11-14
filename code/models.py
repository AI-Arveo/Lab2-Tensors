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

        #Convolutional layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  #First conv layer, 32 filters to find features --> 2^x for memory allocation
            nn.ReLU(),  #Activation to add non-linearity
            nn.MaxPool2d(kernel_size=2, stride=2)  #Pooling layer to reduce size by 2x
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  #Second conv layer with 64 filters
            nn.ReLU(),  #Another ReLU activation
            nn.MaxPool2d(kernel_size=2, stride=2)  #Another pooling layer
        )

        #Fully connected layers
        self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)  #first fully connected layer with 128 neurons
        self.fc2 = nn.Linear(in_features=128, out_features=64)  #second fully connected layer with 64 neurons
        self.fc3 = nn.Linear(in_features=64, out_features=10)  #output layer for 10 classes

    def forward(self, x):
        #Forward pass through conv layers
        x = self.layer1(x)
        x = self.layer2(x)

        #Flatten the output to a 1D vector for fully connected layers
        x = x.view(x.size(0), -1)

        #Passing through fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        #Applying softmax to get class probabilities
        return nn.Softmax(dim=1)(x)

"""This huge comment block contains the code for our Third Iteration Neural Network
    It makes use of Linear Layer functionality and not Conv2D. To test this code you'll have to comment the above NeuralNetworkClassificationModel"""
# class NeuralNetworkClassificationModel(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         """ START TODO: fill in all three layers. Remember that each layer should contain 2 parts, a linear transformation and a nonlinear activation function."""
#         self.layer1 = nn.Sequential(
#             #nn.Linear(in_features=1, out_features=1),  # Klopt dit? Waar vind ik dit?
#             nn.Linear(in_features=784, out_features=256), # activation function
#             nn.ReLU()
#         )
#         self.layer2 = nn.Sequential(
#             nn.Linear(in_features=256, out_features=64),
#             nn.ReLU() #alpha => Default: 1.0
#         )
#
#         self.layer3 = nn.Sequential(
#             nn.Linear(in_features=64, out_features=10),  # assuming 3 classes
#             nn.Softmax(dim=1)  # Softmax along the class dimension -> dimension along which Softmax will be computed(so every slice along dim will sum to 1).
#         )
#         """END TODO"""
#
#     def forward(self, x: torch.Tensor):
#         """START TODO: forward tensor x through all layers."""
#         x = self.layer1(x.flatten(1))
#         x = self.layer2(x)
#         x = self.layer3(x)
#         """END TODO"""
#         return x

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
