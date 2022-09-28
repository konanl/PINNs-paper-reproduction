import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from model import FCNN
import numpy as np
from plotting import Plot
import matplotlib.pyplot as plt
from data_loader import data_generate
from pde import poisson_sol, poisson_sol_gred
import time

#######################################
#   train PINNs & gPINNs poisson 1d   #
#######################################



class Poisson_gpinns(nn.Layer):
    """ PINNs & gPINNs solving 1D Poisson Equations """
    def __init__(self, x, layers, w_g=0):
        """ Initialize the Class """
        super(Poisson_gpinns, self).__init__()
        
        # DATA
        self.x = x
        
        # Full Connected Neural Network
        self.fc_net = FCNN(layers, nn.initializer.XavierUniform())
        
        # Wight of Loss Function
        # L = L_f + w_g * L_g
        self.w_g = w_g
        
        # optimizer
        self.optimizer = optim.Adam(parameters=self.fc_net.parameters())
        
    def forward(self, x):
        """ The Surrogate of the Solution """
        return self.output_transform(x)
    
    def output_transform(self, x):
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
        u = self.forward(x)
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
        # u = self.fc_net(x)
        u = self.forward(x) 
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
    
    def l2_relative_error(self, y_true, y_pred):
        return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
    
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
        
    def test(self):
        pass
        
    
    def predict(self, X):
        """ Predict the u and the gradient of u of the PDE """
        self.fc_net.eval()
        X.stop_gradient = False
        U = self.forward(X)
        dU_dX = paddle.grad(U, X,  retain_graph=False, create_graph=False)[0]
        print(' start predict ... ')
        return U, dU_dX


if __name__ == '__main__':
    
    # DATA
    data_list = data_generate(15, np.pi, 0)
    
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
    
    
    ########################### Plot the Figure.2 D & E ###########################
    
    # figure D
    plot = Plot()
    
    plt.rcParams.update({"font.size": 16})
    plt.figure()
    
    x0 = paddle.reshape(paddle.to_tensor(np.linspace(0, np.pi, 1000)), (1000, 1))
    plot.plot_predict(x0, Poisson_sol(x0), label="Exact", ylabel='u', color="black", linestyle="solid")

    x1 = paddle.reshape(paddle.to_tensor(np.linspace(0, np.pi, 15)), (15, 1))
    plot.plot_predict(x1, Poisson_sol(x1), label=None, ylabel='u', color="black", marker="o", linestyle="none")

    u_pinns, u_pinns_g = model_pinns.predict(paddle.to_tensor(x0, dtype='float32'))
    u_gpinns, u_gpinns_g = model_gpinns.predict(paddle.to_tensor(x0, dtype='float32'))

    plot.plot_predict(x0, u_pinns, label="NN", ylabel='u', color="blue")
    plot.plot_predict(x0, u_gpinns, label="gNN, w = 0.01", ylabel='u', color="red")
    plt.savefig('./result/figure/poisson/u.png')
    
    # figure E
    plt.figure()

    plot.plot_predict(x0, Poisson_sol_gred(x0), label="Exact", ylabel='u`', color="black", linestyle="solid")
    plot.plot_predict(x1, Poisson_sol_gred(x1), label=None, ylabel='u`', color="black", marker="o", linestyle="none")

    plot.plot_predict(x0, u_pinns_g, label="NN", ylabel='u`', color="blue", linestyle="dashed")
    plot.plot_predict(x0, u_gpinns_g, label="gNN, w = 0.01", ylabel='u`', color="red", linestyle="dashed")
    plt.savefig('./result/figure/poisson/u`.png')

    
    ########################### Plot the Figure.2 A、B & C ###########################
    
    # figure A
    training_points = np.linspace(10, 20, 11)
    
    l2_error_u = {}
    l2_error_u_g = {}
    mean_pde_residual = {}
    
    for training_point in training_points:
        
        # DATA
        data_train = data_generate(int(training_point), np.pi, 0)
        
        x_train = paddle.reshape(paddle.to_tensor(data_train, dtype='float32'), [training_point, 1])
        
        print('###### Train PINNs with Training Point: {} ######'.format(training_point))
        model_pinns = (Poisson_gpinns(x, [1, 20, 20, 20, 1]))
        model_pinns.train(20000)
        
        print('###### Train gPINNs w = {} with Training Point: {} ######'.format(0.01, training_point))
        model_gpinns_w_0_01 = (Poisson_gpinns(x, [1, 20, 20, 20, 1], w_g = 0.01))
        model_gpinns_w_0_01.train(20000)
        
        print('###### Train gPINNs w = {} with Training Point: {} ######'.format(1, training_point))
        model_gpinns_w_1 = (Poisson_gpinns(x, [1, 20, 20, 20, 1], w_g = 1))
        model_gpinns_w_1.train(20000)
        
        # save plot data
        data_test = data_generate(100, np.pi, 0)
        
        x_test = paddle.reshape(paddle.to_tensor(data_test, dtype='float32'), [100, 1])
        
        y_true_u, y_true_u_g = Poisson_sol(x_test), Poisson_sol_gred(x_test)
        
        y_pred_u_pinn, y_pred_u_g_pinn = model_pinns.predict(x_test)
        l2_u_pinn = model_pinns.l2_relative_error(y_true_u, y_pred_u_pinn)
        l2_u_g_pinn = model_pinns.l2_relative_error(y_true_u_g, y_pred_u_g_pinn)
        pde_loss_pinn = paddle.sum(model_pinns.pde(x_test)) / 100
        
        y_pred_u_gpinn_w_0_01, y_pred_u_g_gpinn_w_0_01 = model_gpinns_w_0_01.predict(x_test)
        l2_u_gpinn_w_0_01 = model_gpinns_w_0_01.l2_relative_error(y_true_u, y_pred_u_gpinn_w_0_01)
        l2_u_g_gpinn_w_0_01 = model_pinns.l2_relative_error(y_true_u_g, y_pred_u_g_gpinn_w_0_01)
        pde_loss_gpinn_w_0_01 = paddle.sum(model_gpinns_w_0_01.pde(x_test)) / 100
        
        y_pred_u_gpinn_w_1, y_pred_u_g_gpinn_w_1 = model_gpinns_w_1.predict(x_test)
        l2_u_gpinn_w_1 = model_gpinns_w_1.l2_relative_error(y_true_u, y_pred_u_gpinn_w_1)
        l2_u_g_gpinn_w_1 = model_gpinns_w_1.l2_relative_error(y_true_u_g, y_pred_u_g_gpinn_w_1)
        pde_loss_gpinn_w_1 = paddle.sum(model_gpinns_w_1.pde(x_test)) / 100
        
        l2_error_u['training_point_{}'.format(int(training_point))] = [l2_u_pinn, l2_u_gpinn_w_0_01, l2_u_gpinn_w_1]
        l2_error_u_g['training_point_{}'.format(int(training_point))] = [l2_u_g_pinn, l2_u_g_gpinn_w_0_01, l2_u_g_gpinn_w_1]
        mean_pde_residual['training_point_{}'.format(int(training_point))] = [pde_loss_pinn, pde_loss_gpinn_w_0_01, pde_loss_gpinn_w_1]
    
    l2_pinn = []
    l2_gpinn_01 = []
    l2_gpinn_1 = []
    for i in training_points:
        l2_pinn.append(l2_error_u['training_point_{}'.format(int(i))][0])
        l2_gpinn_01.append(l2_error_u['training_point_{}'.format(int(i))][1])
        l2_gpinn_1.append(l2_error_u['training_point_{}'.format(int(i))][2])
        
    # figure A
    plt.figure(dpi=120)
    plt.plot(training_points, l2_pinn, 'o-', color="black", label="PINN")
    plt.plot(training_points, l2_gpinn_01, 's-', color="red", label="gPINN, w=0.01")
    plt.plot(training_points, l2_gpinn_1, '^-', color="blue", label="gPINN, w=1")
    plt.xlabel('No. of training points')
    plt.ylabel('L2 relative error of u')
    plt.legend(frameon=False, loc='best', fontsize=10)
    plt.savefig('./result/figure/poisson/figure2_A.png', dpi=120)
        