import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from model import FCNN
import numpy as np
import matplotlib.pyplot as plt
import sympy
import random
import time
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

#########################################
#    train gPINNs diffusion reaction    #
#########################################

'''
The Partial Differential Equations is:
    
    \frac{\partial u}{\partial t} = D\frac{\partial^2 u}{\partial x^2} + R(x,t)
    
    R(x,t) = e^{-t}[\frac{3}{2}\sin(2x) + \frac{8}{3}\sin(3x) + \frac{15}{4}\sin(4x) + \frac{63}{8}\sin(8x)] 
'''

np.random.seed(142589)

# Analytic Solution
def diffusion_reaction_sol(a):
    """ The Analytic Solution of Diffusion Reaction Equation """
    x, t = a[:, 0:1], a[:, 1:2]
    val = np.sin(8 * x) / 8
    for i in range(1, 5):
        val += np.sin(i * x) / i
    return np.exp(-t) * val


def icfunc(x):
    """ The IC u(x, 0) """
    return (
        paddle.sin(8 * x) / 8
        + paddle.sin(1 * x) / 1
        + paddle.sin(2 * x) / 2
        + paddle.sin(3 * x) / 3
        + paddle.sin(4 * x) / 4
    )


class Diffusion_Reaction_pinns(nn.Layer):
    """ PINNs solving  Diffusion Reaction Equations """
    def __init__(self, X, layers):
        super(Diffusion_Reaction_pinns, self).__init__()
        
        # DATA
        self.x = X[:, 0]
        self.t = X[:, 1]
        
        # Full Connected Neural Network
        self.fc_net = FCNN(layers, None)
        
        # optimizer
        self.optimizer = optim.Adam(learning_rate=0.001, parameters=self.fc_net.parameters())
        
    def forward(self, x, t):
        return self.output_transform(x, t)
    
    def output_transform(self, x, t):
        x_in, t_in = x, t
        return paddle.reshape((x_in - np.pi) * (x_in + np.pi) * (1 - paddle.exp(-t_in)), self.fc_net(X).shape) * self.fc_net(X) + icfunc(x_in)
    
    def diffusion_reaction_sol(self, x, t):
        """ Analytic Solution """
        val = paddle.sin(8 * x) / 8
        for i in range(1, 5):
            val += paddle.sin(i * x) / i
        return paddle.exp(-t) * val
    
    def pde(self, x, t):
        """ loss PDE """
        D = 1
        x.stop_gradient = False
        t.stop_gradient = False
        u = self.forward(x, t)
        du_t = paddle.grad(u, t, retain_graph=True, create_graph=True)[0]
        du_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]
        du_xx = paddle.grad(du_x, x, retain_graph=True, create_graph=True)[0]
        
        r = paddle.exp(-t) * (
            3 * paddle.sin(2 * x) / 2
            + 8 * paddle.sin(3 * x) / 3
            + 15 * paddle.sin(4 * x) / 4
            + 63 * paddle.sin(8 * x) / 8
        )
        
        return du_t -  D * du_xx - r
    
    def loss(self, X):
        """ L = Lf """
        x, t = X[:, 0], X[:, 1]
        loss = paddle.mean(paddle.square(self.pde(x, t))) \
        + paddle.mean(paddle.square(self.fc_net(X) - self.diffusion_reaction_sol(x, t)))
        return loss
    
    def train(self, epochs):
        """ Train PINNs """
        start_time = time.time()
        print(' Start train ... ')
        loss_sum = 0
        loss_alt = []
        for epoch in range(epochs):
            loss = self.loss(X)
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
        """ Predict the u of the PDE """
        self.fc_net.eval()
        U = self.fc_net(X)
        print(' Success Predict ... ')
        return U

    
def gen_test_x(num):
    """ Generate Test Data """
    x = np.linspace(-np.pi, np.pi, num)
    t = np.linspace(0, 1, num)
    l = []

    for i in range(len(t)):
        for j in range(len(x)):
            l.append([x[j], t[i]])
    return np.array(l)
    

class Diffusion_Reaction_gpinns(nn.Layer):
    def __init__(self, X, layers, w_g = 0.001):
        super(Diffusion_Reaction_gpinns, self).__init__()
        
        #DATA
        self.x = X[:, 0]
        self.t = X[:, 1]
        
        # Full Connected Neural Network
        self.fc_net = FCNN(layers, None)
        
        # Optimizer
        self.optimizer = optim.Adam(learning_rate=0.001, parameters=self.fc_net.parameters())
        
        # W_g
        self.w_g = w_g
        
    def forward(self, x, t):
        return self.output_transform(x, t)
    
    def output_transform(self, x, t):
        x_in, t_in = x, t
        return paddle.reshape((x_in - np.pi) * (x_in + np.pi) * (1 - paddle.exp(-t_in)), self.fc_net(X).shape) * self.fc_net(X) + icfunc(x_in)
    
    def diffusion_reaction_sol(self, x, t):
        """ Analytic Solution """
        val = paddle.sin(8 * x) / 8
        for i in range(1, 5):
            val += paddle.sin(i * x) / i
        return paddle.exp(-t) * val
    
    def pde(self, x, t):
        """ loss PDE """
        D = 1
        x.stop_gradient = False
        t.stop_gradient = False
        u = self.forward(x, t)
        du_t = paddle.grad(u, t, retain_graph=True, create_graph=True)[0]
        du_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]
        du_xx = paddle.grad(du_x, x, retain_graph=True, create_graph=True)[0]
        
        r = paddle.exp(-t) * (
            3 * paddle.sin(2 * x) / 2
            + 8 * paddle.sin(3 * x) / 3
            + 15 * paddle.sin(4 * x) / 4
            + 63 * paddle.sin(8 * x) / 8
        )
        
        return du_t -  D * du_xx - r
    
    def pde_g(self, x, t):
        """ loss PDE Gredient """
        x.stop_gradient = False
        t.stop_gradient = False
        du = self.pde(x, t)
        du_x = paddle.grad(du, x, retain_graph=True, create_graph=True)[0]
        du_t = paddle.grad(du, t, retain_graph=True, create_graph=True)[0]
        
        return du_x, du_t
        
#         du_t = paddle.grad(u, t, retain_graph=True, create_graph=True)[0]
#         du_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]
#         du_xx = paddle.grad(du_x, x, retain_graph=True, create_graph=True)[0]
        
#         r = paddle.exp(-t) * (
#             3 * paddle.sin(2 * x) / 2
#             + 8 * paddle.sin(3 * x) / 3
#             + 15 * paddle.sin(4 * x) / 4
#             + 63 * paddle.sin(8 * x) / 8
#         )
        
        
        
#         du_t = paddle.grad(u, t, retain_graph=True, create_graph=True)[0]
#         du_x = paddle.grad(u, x, retain_graph=True, create_graph=True)[0]
#         du_xx = paddle.grad(du_x, x, retain_graph=True, create_graph=True)[0]
        
#         du_tt = paddle.grad(du_t, t, retain_graph=True, create_graph=True)[0]
#         du_xxt = paddle.grad(du_xx, t, retain_graph=True, create_graph=True)[0]
#         dr_t = -r
        
#         du_tx = paddle.grad(du_t, x, retain_graph=True, create_graph=True)[0]
#         du_xxx = paddle.grad(du_xx, x, retain_graph=True, create_graph=True)[0]
#         dr_x = paddle.exp(-t_in) * (
#               63 * paddle.cos(8 * x_in)
#               + 15 * paddle.cos(4 * x_in)
#               + 8 * paddle.cos(3 * x_in)
#               + 3 * paddle.cos(2 * x_in)
#               )
        
        # return du_tt - du_xxt - dr_t, du_tx - du_xxx - dr_x
    
    def loss(self, X):
        """ Loss Function: L = Lf + w_g * Lg """
        x, t = X[:, 0], X[:, 1]
        loss_f = self.pde(x, t)
        loss_g_x, loss_g_t = self.pde_g(x, t)
        loss = paddle.mean(paddle.square(loss_f)) + self.w_g * paddle.mean(paddle.square(loss_g_t)) \
        + self.w_g * paddle.mean(paddle.square(loss_g_x))
        
        return loss
    
    def train(self, epochs):
        start_time = time.time()
        print(' Start train ... ')
        loss_sum = 0
        loss_alt = []
        for epoch in range(epochs):
            loss = self.loss(X)
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
        """ Predict the u of the PDE """
        self.fc_net.eval()
        U = self.fc_net(X)
        print(' Success Predict ... ')
        return U
    
    
    
if __name__ == '__main__':
    
    # DATA
    LARGE_INT = 1000000
    x_list = []
    for i in range(50):
        data_random = random.randint(-LARGE_INT, LARGE_INT)*1.0/LARGE_INT
        if data_random not in x_list:
            data_random *= np.pi # let x -> [-Π,Π]
            x_list.append(data_random)
            
    t_list = []
    for i in range(50):
        data_random = random.randint(0, LARGE_INT)*1.0/LARGE_INT
        if data_random not in t_list:  # t -> [0, 1]
            t_list.append(data_random)
    
    data_list = []
    for i in range(len(x_list)):
        data_list.append([x_list[i], t_list[i]])
        
    X = paddle.reshape(paddle.to_tensor(data_list, dtype='float32'), [50, 2])
    
    # Train with GPU
    paddle.device.set_device('gpu')
    
    # train pinns
    print('Train Success ... ')
    model_pinn = Diffusion_Reaction_pinns(X, [2, 20, 20, 20, 1])
    model_pinn.train(60000)
    
    print('Train Success... ')
    model_gpinn = Diffusion_Reaction_gpinns(X, [2, 20, 20, 20, 1])
    model_gpinn.train(60000)
    
    ########## plot ##########
    plt.rcParams.update({"font.size": 16})
    
    # generate test data
    X = gen_test_x(100)
    x_test = paddle.reshape(paddle.to_tensor(X, dtype='float32'), X.shape)
    
    # pinn
    y_pred_pinn = model_pinn.predict(x_test).tolist()
    
    disp_pinn = []
    prev_pinn = X[0][1]
    temp_pinn = []

    for i in range(len(y_pred_pinn)):
        if X[i][1] == prev_pinn:
            temp_pinn.append(y_pred_pinn[i][0])
        else:

            prev_pinn = X[i][1]

            temp2_pinn = []
            for elem in temp_pinn:
                temp2_pinn.append((elem))
            disp_pinn.append(temp2_pinn)
            temp_pinn.clear()

            temp_pinn.append(y_pred_pinn[i][0])
    
    disp_pinn.reverse()
    plt.figure(figsize=(7, 7))
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(" PINN Prediction ")

    ax = plt.gca()
    im = ax.imshow(disp_pinn, extent=[0, 1, 0, 1])

    ax.set_aspect(1)

    divider = make_axes_locatable(ax)
    width = ax.get_position().width
    height = ax.get_position().height
    cax = divider.append_axes("right", size="5%", pad=0.2)

    plt.colorbar(im, cax=cax)
    plt.savefig('./result/figure/diffusion reaction/u_pinn.png', dpi=120, bbox_inches='tight')
    
    # gpinn
    y_pred_gpinn = model_gpinn.predict(x_test).tolist()
    
    disp_gpinn = []
    prev_gpinn = X[0][1]
    temp_gpinn = []

    for i in range(len(y_pred_gpinn)):
        if X[i][1] == prev_gpinn:
            temp_gpinn.append(y_pred_gpinn[i][0])
        else:

            prev_gpinn = X[i][1]

            temp2_gpinn = []
            for elem in temp_gpinn:
                temp2_gpinn.append((elem))
            disp_gpinn.append(temp2_gpinn)
            temp_gpinn.clear()

            temp_gpinn.append(y_pred_gpinn[i][0])
    
    disp_gpinn.reverse()
    plt.figure(figsize=(7, 7))
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(" gPINN Prediction ")

    ax = plt.gca()
    im = ax.imshow(disp_gpinn, extent=[0, 1, 0, 1])

    ax.set_aspect(1)

    divider = make_axes_locatable(ax)
    width = ax.get_position().width
    height = ax.get_position().height
    cax = divider.append_axes("right", size="5%", pad=0.2)

    plt.colorbar(im, cax=cax)
    plt.savefig('./result/figure/diffusion reaction/u_gpinn.png', dpi=120, bbox_inches='tight')
    
    ##########################
    
