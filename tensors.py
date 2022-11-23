import torch

from options.options import Options
from utilities.utils import plot_tensor, mse, init_pytorch, not_implemented


def create_image(options: Options) -> torch.Tensor:
    """
    TODO: implement this method
    use options to put the tensor to the correct device.
    """

    rgb = [[[0.5021, 0.2843, 0.1935], #given value of lab 2
                [0.8017, 0.5914, 0.7038]],
               [[0.1138, 0.0684, 0.5483],
                [0.8733, 0.6004, 0.5983]],
               [[0.9047, 0.6829, 0.3117],
                [0.6258, 0.2893, 0.9914]]]
    cuda_device = torch.device("CPU" if options.device == "CPU" else "CUDA:0") #chose if we run this algorithm on the CPU or GPU

    image = torch.FloatTensor(rgb) #make the absolute value and return the modified tensor
    image.to(cuda_device)
    print(f"RGB image tensor: \n {image}, with shape:{image.shape}. \n")

    print(f"Plotting this tensor: \n ")

    plot_tensor(image, 'tensor') #will draw the tensor

    return image


def lin_layer_forward(weights: torch.Tensor, random_image: torch.Tensor) -> torch.Tensor:
    """TODO: implement this method"""


    return torch.dot(weights, random_image) #return the values


def tensor_network():
    target = torch.FloatTensor([0.5]).to(options.device)
    print(f"The target is: {target.item():.2f}")
    plot_tensor(target, "Target")

    input_tensor = torch.FloatTensor([0.4, 0.8, 0.5, 0.3]).to(options.device)
    weights = torch.FloatTensor([0.1, -0.5, 0.9, -1]).to(options.device)
    """START TODO:  ensure that the tensor 'weights' saves the computational graph and the gradients after backprop"""
    weights.requires_grad = True
    """END TODO"""

    # remember the activation a of a unit is calculated as follows:
    #      T
    # a = W * x, with W the weights and x the inputs of that unit
    output = lin_layer_forward(weights, input_tensor).to(options.device)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Initial Output")

    # We want a measure of how close we are according to our target
    loss = mse(output, target)
    print(f"The initial loss is: {loss.item():.2f}\n")

    # Lets update the weights now using our loss..
    print(f"The current weights are: {weights}")

    """START TODO: the loss needs to be backpropagated"""
    weights.grad = torch.zeros_like(weights)
    loss.backward(torch.ones_like(loss))
    """END TODO"""

    print(f"The gradients are: {weights.grad}")
    """START TODO: implement the update step with a learning rate of 0.5"""
    # use tensor operations, recall the following formula we've seen during class: x <- x - alpha * x'
    weights = weights - 0.5 *weights.grad*loss
    """END TODO"""
    print(f"The new weights are: {weights}\n")

    # What happens if we forward through our layer again?
    output = lin_layer_forward(weights, input_tensor)
    print(f"Output value: {output.item(): .2f}")
    plot_tensor(output.detach(), "Improved Output")


if __name__ == "__main__":
    options = Options()
    init_pytorch(options)
    tensor_network()
