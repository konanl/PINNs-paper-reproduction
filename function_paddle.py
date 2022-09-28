import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddlescience as psci
import numpy as np
import matplotlib.pyplot as plt
import sympy
import random
import time


###############################
#    train gPINNs function    #
###############################

'''
The Function is:
    
    u(x) = -(1.4 - 3 * x) * sin(18 * x) , x->[0,1]

'''

# Define Analytical solution
def func(x):
    return -(1.4 - 3 * x) * np.sin(18 * x)

def func_grad(x):
    return 3 * np.sin(18 * x) + 18 * (3 * x - 1.4) * np.cos(18 * x)


# Define Full Connection Neural Network
class FCnet(nn.Layer):
    def __init__(self):
        super(FCnet, self).__init__()
        
        # neural network
        layers = []
        layers.append(nn.Linear(1, 20))
        layers.append(nn.Tanh())
        
        for i in range(2):
            layers.append(nn.Linear(20, 20))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(20, 1))
        
        self.FC_net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.FC_net(x)
    

# Define PINNs Using the Frame of PaddlePaddle
class PINNs_func_Paddle(nn.Layer):
    def __init__(self, x):
        super(PINNs_func_Paddle, self).__init__()
        
        # DATA
        self.x = x
        # self.u = u
        
        # full connection neural networks
        self.fc_net = FCnet()
        
        self.optimizer = optim.Adam(parameters=self.fc_net.parameters())
        
    def forward(self):
        self.optimizer.clear_grad()
        u_pred = self.net_u(paddle.to_tensor(self.x, dtype='float32'))
        y_true = self.net_f(paddle.to_tensor(self.x, dtype='float32'))
        
        return u_pred, y_true
    
    def initialize_NN(self, layers):
        pass
    
    def net_u(self, x):
        u = self.fc_net(x)
        return u
    
    def net_f(self, x):# NNfunc()
        f = -(1.4 - 3 * x) * paddle.sin(18 * x)
        return f
    
    def loss_func(self):
        pass
    
    def train(self, nIter):
        
        start_time = time.time()
        print("########## Train PINNs ##########")
        loss_sum = 0
        loss_alt = []
        for i in range(nIter):
            u_pred, f_true = self.forward()
            loss = paddle.mean(paddle.square(u_pred - f_true))
            loss_alt.append([loss])
            loss_sum += loss
            loss.backward()
            self.optimizer.minimize(loss)
            self.optimizer.step()
            self.optimizer.clear_grad()
            if i % 100 == 0:
                elapsed = time.time() - start_time
                loss_mean = loss_sum / (i+1)
                print(
                    'It: %d, Loss: %.3e, Mean Loss: %.3e, Time: %.2f' %
                (
                    i, loss, loss_mean, elapsed
                )
                     )
        mean_loss = np.sum(loss_alt)/(nIter+1)
        print('It: %d, Mean Loss: %.3e' % (nIter, mean_loss))
                            
    def predict(self, X):
        self.fc_net.eval()
        u = self.net_u(X)
        print('u has been predicted!')
        return u
    
    def predict_grad(self, X):
        self.fc_net.eval()
        X.stop_gradient = False
        U = self.net_u(X)
        du_dx = paddle.grad(U, X,  retain_graph=False, create_graph=False)[0]
        print('u` has been predicted!')
        return du_dx


class PINNs_gfunc_Paddle(nn.Layer):
    def __init__(self, x):
        super(PINNs_gfunc_Paddle, self).__init__()
        
        # DATA
        self.x = x
        
        # full connection neural networks
        self.fc_net = FCnet()
        
        # optimizer
        self.optimizer = optim.Adam(parameters=self.fc_net.parameters())
        
    def forward(self, x):
        self.optimizer.clear_grad()
        return self.loss(x)
    
    def loss(self, x):
        w_f = 1
        w_g = 1
        loss = w_f * self.loss_f(x) + w_g * self.loss_g(x)
        return loss
    
    def func(self, x):
        y = -(1.4 - 3 * x) * paddle.sin(18 * x)
        return y
    
    def func_grad(self, x):
        dy_dx = 3 * paddle.sin(18 * x) - 18 * (1.4 - 3 * x) * paddle.cos(18 * x)
        return dy_dx
    
    def loss_f(self, x):
        u_pred = self.fc_net(x)
        loss_f = paddle.mean(paddle.square(u_pred - self.func(x)))
        return loss_f
    
    def loss_g(self, x):
        x.stop_gradient = False
        u_pred = self.fc_net(x)
        du_dx = paddle.grad(u_pred, x, retain_graph=True, create_graph=True)[0]
        dy_dx = self.func_grad(x)
        loss_g = paddle.mean(paddle.square(du_dx - dy_dx))
        return loss_g
    
    def train(self, nIter):
        start_time = time.time()
        print("########## Train gPINNs ##########")
        loss_sum = 0
        loss_alt = []
        for i in range(nIter):
            loss = self.forward(x)
            loss_alt.append([loss])
            loss_sum += loss
            loss.backward()
            self.optimizer.minimize(loss)
            self.optimizer.step()
            self.optimizer.clear_grad()
            if i % 100 == 0:
                elapsed = time.time() - start_time
                loss_mean = loss_sum / (i+1)
                print(
                    'It: %d, Loss: %.3e, Mean Loss: %.3e, Time: %.2f' %
                (
                    i, loss, loss_mean, elapsed
                )
                     )
        mean_loss = np.sum(loss_alt)/(nIter+1)
        print('It: %d, Mean Loss: %.3e' % (nIter, mean_loss)) 
    
    def predict(self, X):
        self.fc_net.eval()
        u = self.fc_net(X)
        print('u has been predicted!')
        return u
    
    def predict_grad(self, X):
        self.fc_net.eval()
        X.stop_gradient = False
        U = self.fc_net(X)
        du_dx = paddle.grad(U, X,  retain_graph=False, create_graph=False)[0]
        print('u` has been predicted!')
        return du_dx


if __name__ == '__main__':
    
    # DATA
    # Randomly Take 15 Training Points Between 0 and 1
    LARGE_INT = 1000000
    data_list = []
    for i in range(30):
        data_random = random.randint(0, LARGE_INT)*1.0/LARGE_INT
        if data_random not in data_list:
            data_list.append(data_random)
            
    x = paddle.reshape(paddle.to_tensor(data_list, dtype='float32'), [30, 1])
    
    # Train on the GPU
    paddle.device.set_device('gpu')
    
    # Train with PINNs
    model_pinn = PINNs_func_Paddle(x)
    model_pinn.train(10000)
    
    # Train with gPINNs
    model_gpinn = PINNs_gfunc_Paddle(x)
    model_gpinn.train(10000)
    
    # Predict
    # a = paddle.to_tensor(0.5, dtype='float32')
    # a_pred_pinn = model_pinn.predict(a)
    # a_pred_gpinn = model_gpinn.predict(a)
    # a_true = func(0.5)
    # print("The True Value of %.3e is %.3e" % (0.5, a_true))
    # print("The Predict of PINNs is: %.3e \nThe Predict of gPINNs is: %.3e" % (a_pred_pinn, a_pred_gpinn))
    
    #####################
    ## plot the figure ##
    #####################
    
    plt.rcParams.update({"font.size": 16})
    
    # ===== figure C =====
    plt.figure()
    # true
    x0 = paddle.reshape(paddle.to_tensor(np.linspace(0, 1, 1000)), (1000, 1))
    plt.plot(x0, func(np.array(x0)), label="Exact", color="black")
    
    x1 = paddle.reshape(paddle.to_tensor(np.linspace(0, 1, 15)), (15, 1))
    plt.plot(x1, func(np.array(x1)), color="black", marker="o", linestyle="none")
    
    # predict
    plt.plot(x0, model_pinn.predict(paddle.to_tensor(x0, dtype='float32')).numpy(),\
             label="NN", color="blue", linestyle="dashed")
    plt.plot(x0, model_gpinn.predict(paddle.to_tensor(x0, dtype='float32')).numpy(),\
             label="gNN", color="red", linestyle="dashed")
    
    # label and others params
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend(frameon=False)
    plt.savefig('./figure/function/u.png')
    plt.show()
    
    # ===== figure D =====
    plt.figure()
    
    # true
    x0_g = paddle.reshape(paddle.to_tensor(np.linspace(0, 1, 1000)), (1000, 1))
    plt.plot(x0_g, func_grad(np.array(x0_g)), label="Exact", color="black")
    
    x1_g = paddle.reshape(paddle.to_tensor(np.linspace(0, 1, 15)), (15, 1))
    plt.plot(x1_g, func_grad(np.array(x1_g)), color="black", marker="o", linestyle="none")
    
    # predict
    plt.plot(x0_g, model_pinn.predict_grad(paddle.to_tensor(x0_g, dtype='float32')).numpy(),\
             label="NN", color="blue", linestyle="dashed")
    plt.plot(x0_g, model_gpinn.predict_grad(paddle.to_tensor(x0_g, dtype='float32')).numpy(),\
         label="gNN", color="red", linestyle="dashed")
    
    # label and others params
    plt.xlabel("x")
    plt.ylabel("u'")
    plt.legend(frameon=False)
    plt.savefig('./figure/function/u`.png')
    plt.show()
