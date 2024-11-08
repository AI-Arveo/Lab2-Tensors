import torch
import torchvision

import matplotlib.pyplot as plt


def plot_tensor(to_plot: torch.Tensor, title: str):
    gray_image_tensor = to_plot.view([1, -1, 1])
    numpy_im = gray_image_tensor.numpy()
    plt.imshow(numpy_im, cmap=plt.get_cmap("GnBu"), interpolation="none", vmin=0, vmax=1)
    plt.title(title)
    plt.show()

def plot_rgb_tensor(to_plot: torch.Tensor, title: str):
    fig = plt.figure()
    plt.imshow(torchvision.transforms.ToPILImage()(to_plot), interpolation="None")
    plt.title(title)
    plt.show()

def mse(input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    return torch.sub(input_tensor,target).square().mean()

def create_image() -> torch.Tensor:
    """TODO: implement this method"""
    weights_only = True
    red_channel = [[0.5021, 0.2843, 0.1935],[0.8017, 0.5914, 0.7038]]
    green_channel = [[0.1138, 0.0684, 0.5483],[0.8733, 0.6004, 0.5983]]
    blue_channel = [[0.9047, 0.6829, 0.3117],[0.6258, 0.2893, 0.9914]]
    rgb = [red_channel,green_channel,blue_channel]
    image = torch.FloatTensor(rgb)
    return image


def lin_layer_forward(weights: torch.Tensor, random_image: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""
    inj = weights.transpose(0,0) * random_image
    return inj


def tensor_network():
    target = torch.FloatTensor([0.5])
    print(f"The target is: {target.item():.2f}")
    plot_tensor(target, "Target")

    input_tensor = torch.FloatTensor([0.4, 0.8, 0.5, 0.3])


    weights = torch.FloatTensor([0.1, -0.5, 0.9, -1])
    """START TODO:  Ensure that the tensor 'weights' saves the computational graph and the gradients after backprop"""
    weights.requires_grad = True
    """END TODO"""

    # remember the activation a of a unit is calculated as follows:
    #      T
    # a = W * x, with W the weights and x the inputs of that unit
    output = lin_layer_forward(weights, input_tensor)
    loss = mse(input_tensor, target)
    print(f"The initial loss is: {loss.item():.2f}\n")
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Initial Output")

    # We want a measure of how close we are according to our target
    loss = mse(output, target)
    print(f"The initial loss is: {loss.item():.2f}\n")

    # Lets update the weights now using our loss..
    print(f"The current weights are: {weights}")

    """START TODO: the loss needs to be backpropagated"""
    
    """END TODO"""

    print(f"The gradients are: {weights.grad}")
    """START TODO: implement the update step with a learning rate of 0.5"""
    
    """END TODO"""
    print(f"The new weights are: {weights}\n")

    # What happens if we forward through our layer again?
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Improved Output")


if __name__ == "__main__":
    tensor_network()
