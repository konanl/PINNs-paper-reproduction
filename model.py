import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from collections import OrderedDict


###########################
## The Network Structure ##
###########################


class FCNN(nn.Layer):
    def __init__(self, layers):
        super(FCNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # layers set
        layer_list = list()
        
        for layer in range(self.depth - 1):
            layer_list.append(nn.Linear(layers[layer], layers[layer+1]))
            layer_list.append(nn.Tanh())
        
        layer_list.append(nn.Linear(layers[-2], layers[-1]))
        
        # net
        self.main = nn.Sequential(*layer_list)
        
    def forward(self, x):
        return self.main(x)
    
def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print(name)
    print("The number of parameters: {}".format(num_params))
    
    
   # TEST
if __name__ == '__main__':
    net = FCNN([1, 20, 20, 20, 20, 1])
    print_network(net, "PINNs")