from abc import ABC

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from collections import OrderedDict


###########################
## The Network Structure ##
###########################


def print_network(model, name):
    """ Print out the network information. """
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))


class FCNet(nn.Layer):
    """ Full Connected Neural Network """

    def __init__(self, layers, nn_init):
        super(FCNet, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # layers set
        layer_list = list()

        for layer in range(self.depth - 1):
            layer_list.append(nn.Linear(layers[layer], layers[layer + 1], weight_attr=nn_init, bias_attr=nn_init))
            layer_list.append(nn.Tanh())

        layer_list.append(nn.Linear(layers[-2], layers[-1], weight_attr=nn_init, bias_attr=nn_init))

        # net
        self.main = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.main(x)


class PhysicsInformedNeuralNetwork(nn.Layer):
    """ The Base Class of Physics Informed Neural Network """

    def __init__(self, net_, pde, solution, optimizer_, w_g=0, output_transform=None):
        super(PhysicsInformedNeuralNetwork, self).__init__()

        # Neural network.
        self.net = net_

        # Partial Differential Equation.
        self.PDE = pde

        # Reference analytical solution.
        self.solution = solution

        # Optimizer
        self.optimizer = optimizer_

        # weight about the gradient-enhanced loss of gPINNs
        self.w_g = w_g

        # Output transform.
        self._output_transform = output_transform

    def forward(self, x):
        if self._output_transform:
            return self._output_transform(self.net(x))
        else:
            return self.net(x)


def gradients(y, x, order=1, create=True):
    """Automatic Differentiation and Compute the Jacobian determinant."""
    if order == 1:
        return paddle.grad(y, x, create_graph=True, retain_graph=True)[0]
    else:
        return paddle.stack([paddle.grad([y[:, i].sum()], [x], create_graph=True, retain_graph=True)[0]
                             for i in range(y.shape[1])], axis=-1)

# def gradients(y, x, order=1, create_graph=True, retain_graph=True):
#     """Automatic Differentiation."""
#     x.stop_gradient = False
#     if order == 1:
#         dy = paddle.grad(y, x, create_graph=create_graph, retain_graph=True)[0]
#         return dy
#     else:
#         return gradients(gradients(y, x), x, order=order - 1)


# TEST
if __name__ == '__main__':
    from pde import function
    from process_data import Data
    net = FCNet([1, 20, 20, 20, 20, 1], nn.initializer.XavierUniform())
    print_network(net, "PINNs")
    optimizer = 'Adam'
    model = PhysicsInformedNeuralNetwork(net, function, optimizer)
    data = Data([13, 2], [0, 1], 'uniform', 100)
    train_data = paddle.to_tensor(data.train_data)
    y_pred = model(train_data)
    print(y_pred)
    print(model.w_g)
    # print(gradients(y_pred, train_data))
