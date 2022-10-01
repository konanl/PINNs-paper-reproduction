import argparse

from mpl_toolkits.axes_grid1 import make_axes_locatable

from model import FCNet, PhysicsInformedNeuralNetwork, gradients
import paddle
import paddle.nn as nn
import numpy as np
from process_data import Data, TimeData
from pde import diffusion_reaction_sol, icfunc, du_x, du_t
from solver import Solver
import os
import matplotlib.pyplot as plt

#########################################
#    train gPINNs diffusion reaction    #
#########################################

'''
The Partial Differential Equations is:
    
    \frac{\partial u}{\partial t} = D\frac{\partial^2 u}{\partial x^2} + R(x,t)
    
    R(x,t) = e^{-t}[\frac{3}{2}\sin(2x) + \frac{8}{3}\sin(3x) + \frac{15}{4}\sin(4x) + \frac{63}{8}\sin(8x)] 
    
'''

# Save the config.
parser = argparse.ArgumentParser()

# Training configurations.
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=100000, help='number of total iterations for training')
parser.add_argument('--resume_epochs', type=int, default=None, help='resume training from this step')
# parser.add_argument('--training_points', type=int, default=None, help='training point in domain or boundary')

# Test configuration.
parser.add_argument('--test_epochs', type=int, default=2000, help='how long we should test model')

# Directories.
parser.add_argument('--model_save_dir', type=str, default='.\\models\\diffusion_reaction')
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

pi = np.pi


#################### PINNs for diffusion reaction ####################


# PINNs & gPINNs.
def PINNpde(x, y):
    x.stop_gradient = False
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]
    dy_a = paddle.grad(y, x, create_graph=True, retain_graph=True)[0]
    dy_x, dy_t = dy_a[:, 0:1], dy_a[:, 1:2]
    Ddy_x = paddle.grad(dy_x, x, create_graph=True, retain_graph=True)[0]
    Ddy_t = paddle.grad(dy_t, x, create_graph=True, retain_graph=True)[0]
    dy_xx = Ddy_x[:, 0:1]

    r = paddle.exp(-t_in) * (
            3 * paddle.sin(2 * x_in) / 2
            + 8 * paddle.sin(3 * x_in) / 3
            + 15 * paddle.sin(4 * x_in) / 4
            + 63 * paddle.sin(8 * x_in) / 8
    )

    return [dy_t - dy_xx - r]


def gPINNpde(x, y):
    x.stop_gradient = False
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    dy_a = paddle.grad(y, x, create_graph=True, retain_graph=True)[0]
    dy_x, dy_t = dy_a[:, 0:1], dy_a[:, 1:2]
    Ddy_x = paddle.grad(dy_x, x, create_graph=True, retain_graph=True)[0]
    Ddy_t = paddle.grad(dy_t, x, create_graph=True, retain_graph=True)[0]
    dy_xx = Ddy_x[:, 0:1]
    r = paddle.exp(-t_in) * (
            3 * paddle.sin(2 * x_in) / 2
            + 8 * paddle.sin(3 * x_in) / 3
            + 15 * paddle.sin(4 * x_in) / 4
            + 63 * paddle.sin(8 * x_in) / 8
    )

    dy_tx = Ddy_t[:, 1:2]
    Ddy_xx = paddle.grad(dy_xx, x, create_graph=True, retain_graph=True)[0]
    dy_xxx = Ddy_xx[:, 0:1]
    dr_x = paddle.exp(-t_in) * (
            63 * paddle.cos(8 * x_in)
            + 15 * paddle.cos(4 * x_in)
            + 8 * paddle.cos(3 * x_in)
            + 3 * paddle.cos(2 * x_in)
    )

    dy_tt = Ddy_t[:, 1:2]
    dy_xxt = Ddy_xx[:, 1:2]
    dr_t = -r

    return [dy_t - dy_xx - r, dy_tx - dy_xxx - dr_x, dy_tt - dy_xxt - dr_t]


def output_transform(x, y):
    x_in = x[:, 0:1]
    t_in = x[:, 1:2]

    return (x_in - np.pi) * (x_in + np.pi) * (1 - paddle.exp(-t_in)) * y + icfunc(x_in)


# Analytical solution.
func = diffusion_reaction_sol

# Data.
data = TimeData([50, 0], [-np.pi, np.pi, 0, 1], 'random', nums_test=100)

# Neural network.
initializer = nn.initializer.XavierUniform()
net = FCNet([2] + [20] * 3 + [1], nn_init=initializer)
optimizer = "Adam"

# PINNs model.
diffusion_reaction_PINNs = PhysicsInformedNeuralNetwork(net,
                                                        PINNpde,
                                                        func, optimizer,
                                                        output_transform=output_transform)

# Train the poisson model with module Solver.
solver_PINNs = Solver(data, diffusion_reaction_PINNs, config)
solver_PINNs.train()

# gPINNs model.
diffusion_reaction_gPINNs = PhysicsInformedNeuralNetwork(net,
                                                         gPINNpde,
                                                         func, optimizer,
                                                         w=[1, 0.01],
                                                         output_transform=output_transform)

# Train the poisson model with module Solver.
solver_gPINNs = Solver(data, diffusion_reaction_gPINNs, config)
solver_gPINNs.train()


# Plot.
# Exact.
def gen_test_x(num):
    """ Generate Test Data """
    x = np.linspace(-np.pi, np.pi, num)
    t = np.linspace(0, 1, num)
    l = []

    for i in range(len(t)):
        for j in range(len(x)):
            l.append([x[j], t[i]])
    return np.array(l)


####################    Plot Figure 4     ####################

plt.rcParams.update({"font.size": 16})


def plot_f4(plot=True):
    if plot:
        # generate test data
        X = gen_test_x(100)
        y = diffusion_reaction_sol(X)

        disp = []
        prev = X[0][1]
        temp = []

        for i in range(len(y)):
            if X[i][1] == prev:
                temp.append(y[i][0])
            else:

                prev = X[i][1]

                temp2 = []
                for elem in temp:
                    temp2.append(elem)
                disp.append(temp2)
                temp.clear()

                temp.append(y[i][0])

        disp.reverse()
        plt.figure(figsize=(7, 7))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title(" Exact Solution ")

        ax = plt.gca()
        im = ax.imshow(disp, extent=[0, 1, -1, 1])

        ax.set_aspect(1)

        divider = make_axes_locatable(ax)
        width = ax.get_position().width
        height = ax.get_position().height
        cax = divider.append_axes("right", size="5%", pad=0.2)

        plt.colorbar(im, cax=cax)
        plt.savefig('./result/figure/diffusion reaction/u_exact.png', dpi=120, bbox_inches='tight')

        # PINNs.
        y_pred_pinn = solver_PINNs.predict(X)[0]

        disp_pinn = []
        prev_pinn = X[0][1]
        temp_pinn = []
        #
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
        im = ax.imshow(disp_pinn, extent=[0, 1, -1, 1])

        ax.set_aspect(1)

        divider = make_axes_locatable(ax)
        width = ax.get_position().width
        height = ax.get_position().height
        cax = divider.append_axes("right", size="5%", pad=0.2)

        plt.colorbar(im, cax=cax)
        plt.savefig('./result/figure/diffusion reaction/u_pinn.png', dpi=120, bbox_inches='tight')

        # gPINNs.
        y_pred_gpinn = solver_gPINNs.predict(X)[0].tolist()

        disp_gpinn = []
        prev_gpinn = X[0][1]
        temp_gpinn = []
        #
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
        im = ax.imshow(disp_gpinn, extent=[0, 1, -1, 1])

        ax.set_aspect(1)

        divider = make_axes_locatable(ax)
        width = ax.get_position().width
        height = ax.get_position().height
        cax = divider.append_axes("right", size="5%", pad=0.2)

        plt.colorbar(im, cax=cax)
        plt.savefig('./result/figure/diffusion reaction/u_gpinn.png', dpi=120, bbox_inches='tight')

        # Absolute error
        # PINNs
        ErrorPINNs = np.abs(y - np.array(y_pred_pinn))
        disp_ep = []
        prev_ep = X[0][1]
        temp_ep = []

        for i in range(len(ErrorPINNs)):
            if X[i][1] == prev_ep:
                temp_ep.append(ErrorPINNs[i][0])
            else:

                prev_ep = X[i][1]

                temp2 = []
                for elem in temp_ep:
                    temp2.append(elem)
                disp_ep.append(temp2)
                temp_ep.clear()

                temp_ep.append(ErrorPINNs[i][0])

        disp_ep.reverse()
        plt.figure(figsize=(7, 7))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title(" Absolution Error of NN ")

        ax = plt.gca()
        im = ax.imshow(disp_ep, extent=[0, 1, -1, 1])

        ax.set_aspect(1)

        divider = make_axes_locatable(ax)
        width = ax.get_position().width
        height = ax.get_position().height
        cax = divider.append_axes("right", size="5%", pad=0.2)

        plt.colorbar(im, cax=cax)
        plt.savefig('./result/figure/diffusion reaction/absolution error_NN.png', dpi=120, bbox_inches='tight')


        # gPINNs
        ErrorgPINNs = np.abs(y - np.array(y_pred_gpinn))
        disp_eg = []
        prev_eg = X[0][1]
        temp_eg = []

        for i in range(len(ErrorgPINNs)):
            if X[i][1] == prev_eg:
                temp_eg.append(ErrorgPINNs[i][0])
            else:

                prev_eg = X[i][1]

                temp2 = []
                for elem in temp_eg:
                    temp2.append(elem)
                disp_eg.append(temp2)
                temp_eg.clear()

                temp_eg.append(ErrorgPINNs[i][0])

        disp_eg.reverse()
        plt.figure(figsize=(7, 7))
        plt.xlabel("t")
        plt.ylabel("x")
        plt.title(" Absolution Error of gNN ")

        ax = plt.gca()
        im = ax.imshow(disp_eg, extent=[0, 1, -1, 1])

        ax.set_aspect(1)

        divider = make_axes_locatable(ax)
        width = ax.get_position().width
        height = ax.get_position().height
        cax = divider.append_axes("right", size="5%", pad=0.2)

        plt.colorbar(im, cax=cax)
        plt.savefig('./result/figure/diffusion reaction/absolution error_gNN.png', dpi=120, bbox_inches='tight')

        plt.show()


def plot_f3(training_points, plot=True):
    if plot:
        l2_u = {}
        l2_dudx = {}
        l2_dudt = {}
        MPDEResidual = {}

        for training_point in training_points:
            training_point = int(training_point)

            print("\n################################################")
            print("#### Start training with training point: {} ####".format(training_point))
            print("################################################\n")

            # DATA.
            data_f3 = TimeData([training_point, 0], [-np.pi, np.pi, 0, 1], 'random', nums_test=10000)

            # Training with different training points.
            # Training PINNs.
            solver_PINNs = Solver(data_f3, diffusion_reaction_PINNs, config)
            solver_PINNs.train()

            # Training gPINNs
            # w = 1
            net_1 = FCNet([2] + [20] * 3 + [1], nn_init=initializer)
            diffusion_reaction_gPINNs_1 = PhysicsInformedNeuralNetwork(
                net_1, gPINNpde, func, optimizer, w=[1, 1],
                output_transform=output_transform
            )
            solver_gPINNs_1 = Solver(data_f3, diffusion_reaction_gPINNs_1, config)
            solver_gPINNs_1.train()

            # w = 0.1
            net_01 = FCNet([2] + [20] * 3 + [1], nn_init=initializer)
            diffusion_reaction_gPINNs_01 = PhysicsInformedNeuralNetwork(
                net_01, gPINNpde, func, optimizer, w=[1, 0.1],
                output_transform=output_transform
            )
            solver_gPINNs_01 = Solver(data_f3, diffusion_reaction_gPINNs_01, config)
            solver_gPINNs_01.train()

            # w = 0.01
            net_001 = FCNet([2] + [20] * 3 + [1], nn_init=initializer)
            diffusion_reaction_gPINNs_001 = PhysicsInformedNeuralNetwork(
                net_001, gPINNpde, func, optimizer, w=[1, 0.01],
                output_transform=output_transform
            )
            solver_gPINNs_001 = Solver(data_f3, diffusion_reaction_gPINNs_001, config)
            solver_gPINNs_001.train()

            # Temp for solve gradient.
            temp = paddle.to_tensor(data_f3.test_data)
            temp.stop_gradient = False

            # Save loss
            y_true_u = diffusion_reaction_sol(temp)
            du_x_true = du_x(temp)
            du_t_true = du_t(temp)

            # PINNs
            y_pred_u_pinn = solver_PINNs.predict(temp)[0]
            du_x_pred_pinn = solver_PINNs.predict(temp)[1][:, 0:1]
            du_t_pred_pinn = solver_PINNs.predict(temp)[1][:, 1:2]

            l2uLoss_pinn = solver_PINNs.l2_relative_error(y_true_u, y_pred_u_pinn)
            l2DuDxLoss_pinn = solver_PINNs.l2_relative_error(du_x_true, du_x_pred_pinn)
            l2DuDtLoss_pinn = solver_PINNs.l2_relative_error(du_t_true, du_t_pred_pinn)
            MPDELoss_pinn = paddle.mean(paddle.abs(diffusion_reaction_PINNs.PDE(temp, y_pred_u_pinn)[0]))

            # gPINNs, w = 1
            y_pred_u_gpinn_1 = solver_gPINNs_1.predict(temp)[0]
            du_x_pred_gpinn_1 = solver_gPINNs_1.predict(temp)[1][:, 0:1]
            du_t_pred_gpinn_1 = solver_gPINNs_1.predict(temp)[1][:, 1:2]
            MPDELoss_gpinn_1 = paddle.mean(paddle.abs(diffusion_reaction_gPINNs_1.PDE(temp, y_pred_u_gpinn_1)[0]))

            l2uLoss_gpinn_1 = solver_gPINNs_1.l2_relative_error(y_true_u, y_pred_u_gpinn_1)
            l2DuDxLoss_gpinn_1 = solver_gPINNs_1.l2_relative_error(du_x_true, du_x_pred_gpinn_1)
            l2DuDtLoss_gpinn_1 = solver_gPINNs_1.l2_relative_error(du_t_true, du_t_pred_gpinn_1)
            MPDELoss_gpinn_1 = paddle.mean(paddle.abs(diffusion_reaction_PINNs.PDE(temp, y_pred_u_gpinn_1)[0]))

            # gPINNs, w = 0.1
            y_pred_u_gpinn_01 = solver_gPINNs_01.predict(temp)[0]
            du_x_pred_gpinn_01 = solver_gPINNs_01.predict(temp)[1][:, 0:1]
            du_t_pred_gpinn_01 = solver_gPINNs_01.predict(temp)[1][:, 1:2]
            MPDELoss_gpinn_01 = paddle.mean(paddle.abs(diffusion_reaction_gPINNs_01.PDE(temp, y_pred_u_gpinn_01)[0]))

            l2uLoss_gpinn_01 = solver_gPINNs_01.l2_relative_error(y_true_u, y_pred_u_gpinn_01)
            l2DuDxLoss_gpinn_01 = solver_gPINNs_01.l2_relative_error(du_x_true, du_x_pred_gpinn_01)
            l2DuDtLoss_gpinn_01 = solver_gPINNs_01.l2_relative_error(du_t_true, du_t_pred_gpinn_01)
            MPDELoss_gpinn_01 = paddle.mean(paddle.abs(diffusion_reaction_PINNs.PDE(temp, y_pred_u_gpinn_01)[0]))

            # gPINNs, w = 0.01
            y_pred_u_gpinn_001 = solver_gPINNs_001.predict(temp)[0]
            du_x_pred_gpinn_001 = solver_gPINNs_001.predict(temp)[1][:, 0:1]
            du_t_pred_gpinn_001 = solver_gPINNs_001.predict(temp)[1][:, 1:2]
            MPDELoss_gpinn_001 = paddle.mean(paddle.abs(diffusion_reaction_gPINNs_001.PDE(temp, y_pred_u_gpinn_001)[0]))

            l2uLoss_gpinn_001 = solver_gPINNs_001.l2_relative_error(y_true_u, y_pred_u_gpinn_001)
            l2DuDxLoss_gpinn_001 = solver_gPINNs_001.l2_relative_error(du_x_true, du_x_pred_gpinn_001)
            l2DuDtLoss_gpinn_001 = solver_gPINNs_001.l2_relative_error(du_t_true, du_t_pred_gpinn_001)
            MPDELoss_gpinn_001 = paddle.mean(paddle.abs(diffusion_reaction_PINNs.PDE(temp, y_pred_u_gpinn_001)[0]))

            # Add loss in dict.
            l2_u['training_point-{}'.format(training_point)] = [l2uLoss_pinn,
                                                                l2uLoss_gpinn_1,
                                                                l2uLoss_gpinn_01,
                                                                l2uLoss_gpinn_001]
            l2_dudx['training_point-{}'.format(training_point)] = [l2DuDxLoss_pinn,
                                                                   l2DuDxLoss_gpinn_1,
                                                                   l2DuDxLoss_gpinn_01,
                                                                   l2DuDxLoss_gpinn_001]
            l2_dudt['training_point-{}'.format(training_point)] = [l2DuDtLoss_pinn,
                                                                   l2DuDtLoss_gpinn_1,
                                                                   l2DuDtLoss_gpinn_01,
                                                                   l2DuDtLoss_gpinn_001]
            MPDEResidual['training_point-{}'.format(training_point)] = [MPDELoss_pinn,
                                                                        MPDELoss_gpinn_1,
                                                                        MPDELoss_gpinn_01,
                                                                        MPDELoss_gpinn_001]
        # Add loss in different PINNs loss.
        l2_u_pinn = []
        l2_u_gpinn_1 = []
        l2_u_gpinn_01 = []
        l2_u_gpinn_001 = []

        l2DuDx_pinn = []
        l2DuDx_gpinn_1 = []
        l2DuDx_gpinn_01 = []
        l2DuDx_gpinn_001 = []

        l2DuDt_pinn = []
        l2DuDt_gpinn_1 = []
        l2DuDt_gpinn_01 = []
        l2DuDt_gpinn_001 = []

        mPDELoss_pinn = []
        mPDELoss_gpinn_1 = []
        mPDELoss_gpinn_01 = []
        mPDELoss_gpinn_001 = []

        for i in training_points:
            i = int(i)
            l2_u_pinn.append(l2_u['training_point-{}'.format(int(i))][0])
            l2_u_gpinn_1.append(l2_u['training_point-{}'.format(int(i))][1])
            l2_u_gpinn_01.append(l2_u['training_point-{}'.format(int(i))][2])
            l2_u_gpinn_001.append(l2_u['training_point-{}'.format(int(i))][3])

            l2DuDx_pinn.append(l2_dudx['training_point-{}'.format(int(i))][0])
            l2DuDx_gpinn_1.append(l2_dudx['training_point-{}'.format(int(i))][1])
            l2DuDx_gpinn_01.append(l2_dudx['training_point-{}'.format(int(i))][2])
            l2DuDx_gpinn_001.append(l2_dudx['training_point-{}'.format(int(i))][3])

            l2DuDt_pinn.append(l2_dudt['training_point-{}'.format(int(i))][0])
            l2DuDt_gpinn_1.append(l2_dudt['training_point-{}'.format(int(i))][1])
            l2DuDt_gpinn_01.append(l2_dudt['training_point-{}'.format(int(i))][2])
            l2DuDt_gpinn_001.append(l2_dudt['training_point-{}'.format(int(i))][3])

            mPDELoss_pinn.append(MPDEResidual['training_point-{}'.format(int(i))][0])
            mPDELoss_gpinn_1.append(MPDEResidual['training_point-{}'.format(int(i))][1])
            mPDELoss_gpinn_01.append(MPDEResidual['training_point-{}'.format(int(i))][2])
            mPDELoss_gpinn_001.append(MPDEResidual['training_point-{}'.format(int(i))][3])

        # Figure A
        plt.figure(dpi=120)
        plt.plot(training_points, l2_u_pinn, 'o-', color='black', label='NN')
        plt.plot(training_points, l2_u_gpinn_1, 's-', color='yellow', label='gNN, w = 1')
        plt.plot(training_points, l2_u_gpinn_01, '^-', color='red', label='gNN, w = 0.1')
        plt.plot(training_points, l2_u_gpinn_001, 'D-', color='blue', label='gNN, w = 0.01')
        plt.xlabel('No. of training points')
        plt.ylabel('$L^2$ relative error of u')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\diffusion reaction\\figure3_A.png', dpi=120)

        # Figure C
        plt.figure(dpi=120)
        plt.plot(training_points, l2DuDx_pinn, 'o-', color='black', label='NN')
        plt.plot(training_points, l2DuDx_gpinn_1, 's-', color='yellow', label='gNN, w = 1')
        plt.plot(training_points, l2DuDx_gpinn_01, '^-', color='red', label='gNN, w = 0.1')
        plt.plot(training_points, l2DuDx_gpinn_001, 'D-', color='blue', label='gNN, w = 0.01')
        plt.xlabel('No. of training points')
        plt.ylabel('$L^2$ relative error of du/dx')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\diffusion reaction\\figure3_C.png', dpi=120)

        # Figure D
        plt.figure(dpi=120)
        plt.plot(training_points, l2DuDt_pinn, 'o-', color='black', label='NN')
        plt.plot(training_points, l2DuDt_gpinn_1, 's-', color='yellow', label='gNN, w = 1')
        plt.plot(training_points, l2DuDt_gpinn_01, '^-', color='red', label='gNN, w = 0.1')
        plt.plot(training_points, l2DuDt_gpinn_001, '^-', color='blue', label='gNN, w = 0.01')
        plt.xlabel('No. of training points')
        plt.ylabel('$L^2$ relative error of du/dx')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\diffusion reaction\\figure2_C.png', dpi=120)

        # Figure B
        plt.figure(dpi=120)
        plt.plot(training_points, mPDELoss_pinn, 'o-', color='black', label='NN')
        plt.plot(training_points, mPDELoss_gpinn_1, 's-', color='yellow', label='gNN, w = 1')
        plt.plot(training_points, mPDELoss_gpinn_01, '^-', color='red', label='gNN, w = 0.1')
        plt.plot(training_points, mPDELoss_gpinn_001, 'D-', color='blue', label='gNN, w = 0.01')
        plt.xlabel('No. of training points')
        plt.ylabel('Mean pde residual')
        plt.legend(frameon=False, loc='best', fontsize=10)
        plt.savefig('.\\result\\figure\\diffusion reaction\\figure3_B.png', dpi=120)
        plt.show()


plot_f4(False)
training_points = [25, 30, 40, 50, 65, 80, 100, 120, 150]
plot_f3(training_points, True)
