import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from model import FCNN
import paddlescience as psci
import numpy as np
import matplotlib.pyplot as plt
import sympy
import random
import time


#######################################
#   train PINNs & gPINNs poisson 1d   #
#######################################


'''
The Partial Differential Function is:
    
    -\Delta u = \sum_{i=1}^{4} i\sin ix + 8\sin 8x, x\in [0, \pi ]

'''

# Analytical solution
def Poisson_sol(x):
    """ The Analytical Solution of Poisson Eqution"""
    sol = x + 1 / 8 * paddle.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * paddle.sin(i * x)
    return sol

def Poisson_sol_gred(x):
    """ The Gredient of Poisson Eqution`s Analytical Solution """
    du = 1 + paddle.cos(8 * x)
    for i in range(1, 5):
        du += paddle.cos(i * x)
    return du

class Poisson_gpinns(nn.Layer):
    """ PINNs & gPINNs solving 1D Poisson Equations """
    def __init__(self, x, layers, w_g=0):
        """ Initialize the Class """
        super(Poisson_gpinns, self).__init__()
        
        # DATA
        self.x = x
        
        # Full Connected Neural Network
        self.fc_net = FCNN(layers)
        
        # Wight of Loss Function
        # L = L_f + w_g * L_g
        self.w_g = w_g
        
        # optimizer
        self.optimizer = optim.Adam(parameters=self.fc_net.parameters())
        
    def forward(self, x):
        """ The Surrogate of the Solution """
        return self._output_transform(x)
    
    def _output_transform(self, x):
        """ The Transform: u(x) = x(\pi - x)\mathcal{N}(x) + x """
        return x + paddle.tanh(x) * paddle.tanh(np.pi - x) * self.fc_net(x)
    
    def poisson_sol(self, x):
        sol = x + 1 / 8 * paddle.sin(8 * x)
        for i in range(1, 5):
            sol += 1 / i * paddle.sin(i * x)
        return sol
    
    def poisson_sol_gred(self, x):
        """ The Gredient of Poisson Eqution`s Analytical Solution """
        dsol = 1 + paddle.cos(8 * x)
        for i in range(1, 5):
            dsol += paddle.cos(i * x)
        return dsol
    
    def net_gred(self, x):
        x.stop_gradient = False
        u = self.fc_net(x)
        du = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]
        return du
    
    def loss(self, x):
        """ L = L_f + w_g * L_g """
        loss = paddle.mean(paddle.square(self.pde(x))) + self.w_g * paddle.mean(paddle.square(self.pde_g(x))) \
        + paddle.mean(paddle.square(self.fc_net(x) - self.poisson_sol(x))) \
        + paddle.mean(paddle.square(self.net_gred(x) - self.poisson_sol_gred(x)))
        return loss
    
    def pde(self, x):
        """ loss PDE """
        x.stop_gradient = False
        u = self.fc_net(x)
        du_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]
        du_xx = paddle.grad(du_x, x, retain_graph=True, create_graph=True)[0]
        
        f = 8 * paddle.sin(8 * x)
        for i in range(1, 5):
            f += i * paddle.sin(i * x)
        
        return - du_xx - f
    
    def pde_g(self, x):
        """ The gradient of PDE """
        x.stop_gradient = False
        # u = self.fc_net(x)
        du_xx = self.pde(x)
        du_xxx = paddle.grad(du_xx, x, retain_graph=True, create_graph=True)[0]
            
        return  - du_xxx
    
    def train(self, epochs):
        """ Train """
        start_time = time.time()
        print(' start train ... ')
        loss_sum = 0
        loss_alt = []
        for epoch in range(epochs):
            loss = self.loss(x)
            loss_alt.append([loss])
            loss_sum += loss
            loss.backward()
            self.optimizer.minimize(loss)
            self.optimizer.step()
            self.optimizer.clear_grad()
            if epoch % 1000 == 0:
                elapsed = time.time() - start_time
                loss_mean = loss_sum / (epoch+1)
                print(
                    'Epoch: %d, Loss: %.3e, Mean Loss: %.3e, Time: %.2f' %
                (
                    epoch, loss, loss_mean, elapsed
                )
                     )
        mean_loss = np.sum(loss_alt) / (epochs+1)
        print('Epoch: %d, Mean Loss: %.3e' % (epochs, mean_loss))  
        
    
    def predict(self, X):
        """ Predict the u and the gradient of u of the PDE """
        self.fc_net.eval()
        X.stop_gradient = False
        U = self.fc_net(X)
        dU_dX = paddle.grad(U, X,  retain_graph=False, create_graph=False)[0]
        print(' start predict ... ')
        return U, dU_dX


if __name__ == '__main__':
    
    # DATA
    LARGE_INT = 1000000
    data_list = []
    for i in range(15):
        data_random = random.randint(0, LARGE_INT)*1.0/LARGE_INT
        if data_random not in data_list:
            data_random *= np.pi # let x -> [0,Π]
            data_list.append(data_random)
            
    x = paddle.to_tensor(data_list, dtype='float32')
    x = paddle.reshape(x, [15,1])
    
    # Train with GPU
    paddle.device.set_device('gpu')
    
    # train pinns
    print(' #################### Train PINNs #################### ')
    model_pinns = (Poisson_gpinns(x, [1, 20, 20, 20, 1]))
    model_pinns.train(20000)
    
    # train gpinns
    print(' #################### Train gPINNs #################### ')
    model_gpinns = (Poisson_gpinns(x, [1, 20, 20, 20, 1], w_g = 0.01))
    model_gpinns.train(20000)
    
    # plot
    plt.rcParams.update({"font.size": 16})

    plt.figure()

    x0 = paddle.reshape(paddle.to_tensor(np.linspace(0, np.pi, 1000)), (1000, 1))
    plt.plot(x0, Poisson_sol(x0), label="Exact", color="black")

    x1 = paddle.reshape(paddle.to_tensor(np.linspace(0, np.pi, 15)), (15, 1))
    plt.plot(x1, Poisson_sol(x1), color="black", marker="o", linestyle="none")

    u_pinns, u_pinns_g = model_pinns.predict(paddle.to_tensor(x0, dtype='float32'))
    u_gpinns, u_gpinns_g = model_gpinns.predict(paddle.to_tensor(x0, dtype='float32'))

    plt.plot(x0, u_pinns, label="NN", color="blue", linestyle="dashed")
    plt.plot(x0, u_gpinns, label="gNN, w = 0.01", color="red", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(frameon=False)
    plt.savefig('./result/poisson/u.png')

    ######################################################################
    
    plt.figure()

    plt.plot(x0, Poisson_sol_gred(x0), label="Exact", color="black")

    # x1_ = paddle.reshape(paddle.to_tensor(np.linspace(0, np.pi, 15)), (15, 1))
    plt.plot(x1, Poisson_sol_gred(x1), color="black", marker="o", linestyle="none")

    plt.plot(x0, u_pinns_g, label="NN", color="blue", linestyle="dashed")
    plt.plot(x0, u_gpinns_g, label="gNN, w = 0.01", color="red", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u`")
    plt.legend(frameon=False)
    plt.savefig('./result/poisson/u`.png')

    plt.show()