import argparse
from model import FCNet, PhysicsInformedNeuralNetwork, gradients
import paddle
import paddle.nn as nn
import numpy as np
from process_data import Data
from pde import poisson_sol, poisson_sol_grad
from solver import Solver
import os
import matplotlib.pyplot as plt


#######################################
#   train PINNs & gPINNs poisson 1d   #
#######################################

"""
    The task is one dimension poisson equation.

"""

# Save the config.
parser = argparse.ArgumentParser()

# Training configurations.
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=20000, help='number of total iterations for training')
parser.add_argument('--resume_epochs', type=int, default=None, help='resume training from this step')
parser.add_argument('--training_points', type=int, default=None, help='training point in domain or boundary')

# Test configuration.
parser.add_argument('--test_epochs', type=int, default=2000, help='how long we should test model')

# Directories.
parser.add_argument('--model_save_dir', type=str, default='.\\models\\poisson_1d')
parser.add_argument('--result_dir', type=str, default='.\\result')

# Step size.
parser.add_argument('--log_step', type=int, default=2000)
parser.add_argument('--model_save_step', type=int, default=2000)

config = parser.parse_args()
print(config)

# Create directories if not exist.
if not os.path.exists(config.model_save_dir):
    os.makedirs(config.model_save_dir)

if not os.path.exists(config.result_dir):
    os.makedirs(config.result_dir)


#################### PINNs for 1d poisson ####################

# PDE loss.
def NNPoisson(x, y):
    """Poisson PDE equation for PINNs."""
    x.stop_gradient = False
    dy_xx = gradients(y, x, order=2)
    f = 8 * paddle.sin(8 * x)
    for i in range(1, 5):
        f += i * paddle.sin(i * x)
    return [-dy_xx - f]


def output_transform(x, y):
    return x + paddle.tanh(x) * paddle.tanh(np.pi - x) * y


# Analytical solution.
func, func_grad = poisson_sol, poisson_sol_grad

# Data.
data = Data([15, 0], [0, np.pi], "uniform", nums_test=100)

# Neural network.
initializer = nn.initializer.XavierUniform()
net_pinns = FCNet([1] + [20] * 3 + [1], nn_init=initializer)
optimizer = "Adam"

# PINNs model.
poisson_PINNs = PhysicsInformedNeuralNetwork(net_pinns, NNPoisson, func, optimizer, output_transform=output_transform)

# Train the poisson model with module Solver
solver_PINNs = Solver(data, poisson_PINNs, config)

# Train.
# solver_PINNs.train()


#################### gPINNs for 1d poisson ####################

# PDE loss.
def gNNPoisson(x, y):
    """Poisson PDE equation for gPINNs."""
    dy_xx = gradients(y, x, order=2)
    f = 8 * paddle.sin(8 * x)
    for i in range(1, 5):
        f += i * paddle.sin(i * x)

    dy_xxx = gradients(y, x, order=3)
    df_x = (
        paddle.cos(x)
        + 4 * paddle.cos(2 * x)
        + 9 * paddle.cos(3 * x)
        + 16 * paddle.cos(4 * x)
        + 64 * paddle.cos(8 * x)
    )
    return [-dy_xx - f, -dy_xxx - df_x]


# Neural Network.
net_gpinns = FCNet([1] + [20] * 3 + [1], nn_init=initializer)

# gPINNs model.
poisson_gPINNs = PhysicsInformedNeuralNetwork(
    net_gpinns, gNNPoisson, func, optimizer,
    w=[1, 0.01], output_transform=output_transform)
solver_gPINNs = Solver(data, poisson_gPINNs, config)

# Train
# solver_gPINNs.train()

####################    Plot Figure 2     ####################
plt.rcParams.update({"font.size": 16})

#################### Plot Figure 2. D & E ####################


def plot_de(plot=True):
    if plot:
        # Figure D.
        plt.figure()

        # Train.
        # solver_PINNs.train()
        # solver_gPINNs.train()

        # True.
        x0 = np.reshape(np.linspace(0, np.pi, 1000), (1000, 1))
        plt.plot(x0, func(x0), label="Exact", color="black")

        x1 = np.reshape(np.linspace(0, np.pi, 15), (15, 1))
        plt.plot(x1, func(x1), color="black", marker="o", linestyle="none")

        # Predict.
        # PINNs.
        u_pred_pinns = solver_PINNs.predict(paddle.to_tensor(x0, dtype='float32'))[0].numpy()
        plt.plot(x0, u_pred_pinns, label="NN", color="blue", linestyle="dashed")
        # gPINNs.
        u_pred_gpinns = solver_gPINNs.predict(paddle.to_tensor(x0, dtype='float32'))[0].numpy()
        plt.plot(x0, u_pred_gpinns, label="gNN, w = 0.01", color="red", linestyle="dashed")

        # label and others params.
        plt.xlabel("x")
        plt.ylabel("u")
        plt.legend(frameon=False)
        plt.savefig('.\\result\\figure\\poisson\\u.png', dpi=120)

        # Figure E.
        plt.figure()

        # True.
        plt.plot(x0, func_grad(x0), label="Exact", color="black")
        plt.plot(x1, func_grad(x1), color="black", marker="o", linestyle="none")

        u_g_pred_pinns = solver_PINNs.predict(paddle.to_tensor(x0, dtype='float32'))[1].numpy()
        plt.plot(x0, u_g_pred_pinns, label="NN", color="blue", linestyle="dashed")

        u_g_pred_gpinns = solver_gPINNs.predict(paddle.to_tensor(x0, dtype='float32'))[1].numpy()
        plt.plot(x0, u_g_pred_gpinns, label="gNN, w = 0.01", color="red", linestyle="dashed")

        # label and others params
        plt.xlabel("x")
        plt.ylabel("u`")
        plt.legend(frameon=False)
        plt.savefig('.\\result\\figure\\poisson\\u`.png', dpi=120)
        plt.show()

#################### Plot Figure 2. A. B. & C. ####################


def plot_abc(plot=True):
    if plot:
        training_points = np.linspace(10, 20, 11)

        l2_error_u = {}
        l2_error_u_g = {}
        mean_pde_residual = {}

        for training_point in training_points:
            training_point = int(training_point)

            print("\n################################################")
            print("#### Start training with training point: {} ####".format(training_point))
            print("################################################\n")

            # DATA.
            data_f_abc = Data([training_point - 2, 2], [0, np.pi], 'uniform', 100)

            # Training with different training points.
            # Training PINNs.
            solver_PINNs = Solver(data_f_abc, poisson_PINNs, config)
            solver_PINNs.train()

            # Training gPINNs.
            # w = 1.
            poisson_gPINNs_1 = PhysicsInformedNeuralNetwork(
                net_gpinns, gNNPoisson, func, optimizer,
                w=[1, 1], output_transform=output_transform)
            solver_gPINNs_w_1 = Solver(data_f_abc, poisson_gPINNs_1, config)
            solver_gPINNs_w_1.train()

            # w = 0.01.
            solver_gPINNs = Solver(data_f_abc, poisson_gPINNs, config)
            solver_gPINNs.train()

            # Temp for solve gradient.
            temp = paddle.to_tensor(data_f_abc.test_data)
            temp.stop_gradient = False

            # Save loss.
            y_true_u, y_true_u_g = poisson_sol(temp), poisson_sol_grad(temp)

            # PINNs.
            y_pred_u_pinns, y_pred_u_g_pinns = solver_PINNs.predict(temp)[0], \
                                               solver_PINNs.predict(temp)[1]
            l2_u_pinn = solver_PINNs.l2_relative_error(y_true_u, y_pred_u_pinns)
            l2_u_g_pinn = solver_PINNs.l2_relative_error(y_true_u_g, y_pred_u_g_pinns)
            mean_pde_loss_pinn = paddle.sum(paddle.abs(poisson_PINNs.PDE(temp, y_pred_u_pinns)[0])) / len(temp)

            # gPINNs, w = 1.
            y_pred_u_gpinns_1, y_pred_u_g_gpinns_1 = solver_gPINNs_w_1.predict(temp)[0], \
                                                     solver_gPINNs_w_1.predict(temp)[1]
            l2_u_gpinn_1 = solver_gPINNs_w_1.l2_relative_error(y_true_u, y_pred_u_gpinns_1)
            l2_u_g_gpinn_1 = solver_gPINNs_w_1.l2_relative_error(y_true_u_g, y_pred_u_g_gpinns_1)
            mean_pde_loss_gpinn_1 = paddle.sum(paddle.abs(poisson_gPINNs_1.PDE(temp, y_pred_u_gpinns_1)[0])) / len(temp)

            # gPINNs, w = 0.01.
            y_pred_u_gpinns, y_pred_u_g_gpinns = solver_gPINNs.predict(temp)[0], \
                                                     solver_gPINNs.predict(temp)[1]
            l2_u_gpinn = solver_gPINNs.l2_relative_error(y_true_u, y_pred_u_gpinns)
            l2_u_g_gpinn = solver_gPINNs_w_1.l2_relative_error(y_true_u_g, y_pred_u_g_gpinns)
            mean_pde_loss_gpinn = paddle.sum(paddle.abs(poisson_gPINNs.PDE(temp, y_pred_u_gpinns)[0])) / len(temp)

            # Add loss in dict.
            l2_error_u['training_point-{}'.format(training_point)] = [l2_u_pinn, l2_u_gpinn_1, l2_u_gpinn]
            l2_error_u_g['training_point-{}'.format(training_point)] = [l2_u_g_pinn, l2_u_g_gpinn_1, l2_u_g_gpinn]
            mean_pde_residual['training_point-{}'.format(training_point)] = [
                mean_pde_loss_pinn,
                mean_pde_loss_gpinn_1,
                mean_pde_loss_gpinn]

        # Add loss in different PINNs loss.
        l2_pinn_u = []
        l2_gpinn_u_1 = []
        l2_gpinn_u = []

        l2_pinn_u_g = []
        l2_gpinn_u_g_1 = []
        l2_gpinn_u_g = []

        m_pde_r_pinn = []
        m_pde_r_gpinn_1 = []
        m_pde_r_gpinn = []

        for i in training_points:
            l2_pinn_u.append(l2_error_u['training_point-{}'.format(int(i))][0])
            l2_gpinn_u_1.append(l2_error_u['training_point-{}'.format(int(i))][1])
            l2_gpinn_u.append(l2_error_u['training_point-{}'.format(int(i))][2])

            l2_pinn_u_g.append(l2_error_u_g['training_point-{}'.format(int(i))][0])
            l2_gpinn_u_g_1.append(l2_error_u_g['training_point-{}'.format(int(i))][1])
            l2_gpinn_u_g.append(l2_error_u_g['training_point-{}'.format(int(i))][2])

            m_pde_r_pinn.append(mean_pde_residual['training_point-{}'.format(int(i))][0])
            m_pde_r_gpinn_1.append(mean_pde_residual['training_point-{}'.format(int(i))][1])
            m_pde_r_gpinn.append(mean_pde_residual['training_point-{}'.format(int(i))][2])

        # Figure A
        plt.figure(dpi=120)
        plt.plot(training_points, l2_pinn_u, 'o-', color='black', label='NN')
        plt.plot(training_points, l2_gpinn_u, 's-', color='red', label='gNN, w = 0.01')
        plt.plot(training_points, l2_gpinn_u_1, '^-', color='blue', label='gNN, w = 1')
        plt.xlabel('No. of training points')
        plt.ylabel(r'L^2'' relative error of u')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\poisson\\figure2_A.png', dpi=120)

        # Figure B
        plt.figure(dpi=120)
        plt.plot(training_points, l2_pinn_u_g, 'o-', color='black', label='NN')
        plt.plot(training_points, l2_gpinn_u_g, 's-', color='red', label='gNN, w = 0.01')
        plt.plot(training_points, l2_gpinn_u_g_1, '^-', color='blue', label='gNN, w = 1')
        plt.xlabel('No. of training points')
        plt.ylabel(r'L^2'' relative error of u`')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\poisson\\figure2_B.png', dpi=120)

        # Figure C
        plt.figure(dpi=120)
        plt.plot(training_points, m_pde_r_pinn, 'o-', color='black', label='NN')
        plt.plot(training_points, m_pde_r_gpinn, 's-', color='red', label='gNN, w = 0.01')
        plt.plot(training_points, m_pde_r_gpinn_1, '^-', color='blue', label='gNN, w = 1')
        plt.xlabel('No. of training points')
        plt.ylabel('Mean pde residual')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\poisson\\figure2_C.png', dpi=120)
        plt.show()

#################### Plot Figure 2. F. & G. ####################


def plot_fg(plot=True):
    if plot:
        data_f_fg = Data([18, 2], [0, np.pi], 'uniform', 100)

        W = [10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 1]

        l2_u = {}
        l2_u_g = {}

        for w in W:

            # PiNNs.
            solver_pinn = Solver(data_f_fg, poisson_PINNs, config)
            # gPINNs.
            poisson_gPINNs_ = PhysicsInformedNeuralNetwork(
                net_gpinns, gNNPoisson, func, optimizer,
                w=[1, w], output_transform=output_transform)
            solver_gpinn = Solver(data, poisson_gPINNs_, config)

            # Train.
            solver_pinn.train()
            solver_gpinn.train()

            # Temp for solve gradient.
            temp = paddle.to_tensor(data_f_fg.test_data)
            temp.stop_gradient = False

            # Save loss.
            y_true_u, y_true_u_g = poisson_sol(temp), poisson_sol_grad(temp)

            # PINNs.
            y_pred_u_pinns, y_pred_u_g_pinns = solver_pinn.predict(temp)[0], \
                                               solver_pinn.predict(temp)[1]
            l2_u_pinn = solver_pinn.l2_relative_error(y_true_u, y_pred_u_pinns)
            l2_u_g_pinn = solver_pinn.l2_relative_error(y_true_u_g, y_pred_u_g_pinns)

            # gPINNs, W = w.
            y_pred_u_gpinns, y_pred_u_g_gpinns = solver_gpinn.predict(temp)[0], \
                                                     solver_gpinn.predict(temp)[1]
            l2_u_gpinn = solver_gpinn.l2_relative_error(y_true_u, y_pred_u_gpinns)
            l2_u_g_gpinn = solver_gpinn.l2_relative_error(y_true_u_g, y_pred_u_g_gpinns)

            # Add loss in dict.
            l2_u['w-{}'.format(w)] = [l2_u_pinn, l2_u_gpinn]
            l2_u_g['w-{}'.format(w)] = [l2_u_g_pinn, l2_u_g_gpinn]

        # Add loss in different PINNs loss.
        l2_u_nn = []
        l2_u_gnn = []

        l2_u_g_nn = []
        l2_u_g_gnn = []

        for w in W:
            l2_u_nn.append(l2_u['w-{}'.format(w)][0])
            l2_u_gnn.append(l2_u['w-{}'.format(w)][1])

            l2_u_g_nn.append(l2_u_g['w-{}'.format(w)][0])
            l2_u_g_gnn.append(l2_u_g['w-{}'.format(w)][1])

        # Figure F
        plt.figure(dpi=120)
        plt.plot(W, l2_u_nn, color='black', label='PINN')
        plt.plot(W, l2_u_gnn, 's-', color='red', label='gPINN')
        plt.xlabel('w')
        plt.ylabel(r'L^2 relative error of u')
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\poisson\\figure2_F.png', dpi=120)

        # Figure G
        plt.figure(dpi=120)
        plt.plot(W, l2_u_g_nn, color='black', label='PINN')
        plt.plot(W, l2_u_g_gnn, 's-', color='red', label='gPINN')
        plt.xlabel('w')
        plt.ylabel(r'L^2 relative error of u`')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='x')
        plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='y')
        plt.savefig('.\\result\\figure\\poisson\\figure2_G.png', dpi=120)
        plt.show()


# Plot all figure

plot_abc(True)
plot_de(False)
plot_fg(False)
