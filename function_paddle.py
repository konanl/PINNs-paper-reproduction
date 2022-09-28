#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gPINNs_re 
@File    ：function_paddle.py
@Author  ：LiangL. Yan
@Date    ：2022/9/26 17:23 
"""

import argparse
from model import FCNet, PhysicsInformedNeuralNetwork, gradients
import paddle
import paddle.nn as nn
import numpy as np
from process_data import Data
from pde import function, function_grad
from solver import Solver
import os
import matplotlib.pyplot as plt

############################
## Function Approximation ##
############################


"""
    The task is function approximation.

    The  Function is:

        u(x) = -(1.4 - 3 * x) * sin(18 * x) , x->[0,1] 

"""

# Save the config
parser = argparse.ArgumentParser()

# Training configurations.
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--num_epochs', type=int, default=10000, help='number of total iterations for training')
parser.add_argument('--resume_epochs', type=int, default=None, help='resume training from this step')
parser.add_argument('--training_points', type=int, default=None, help='training point in domain or boundary')

# Test configuration.
parser.add_argument('--test_epochs', type=int, default=1000, help='how long we should test model')

# Directories.
parser.add_argument('--model_save_dir', type=str, default='.\\models')
parser.add_argument('--result_dir', type=str, default='.\\results')

# Step size.
parser.add_argument('--log_step', type=int, default=1000)
parser.add_argument('--model_save_step', type=int, default=1000)

config = parser.parse_args()
print(config)

# Create directories if not exist.
# if not os.path.exists(config.log_dir):
#     os.makedirs(config.log_dir)
if not os.path.exists(config.model_save_dir):
    os.makedirs(config.model_save_dir)
# if not os.path.exists(config.sample_dir):
#     os.makedirs(config.sample_dir)
if not os.path.exists(config.result_dir):
    os.makedirs(config.result_dir)


#################### PINNs ####################
# Pde loss.
def NNFunc(x, y):
    return [y + (1.4 - 3 * x) * paddle.sin(18 * x)]


# Analytical solution
# def func(x):
#     return -(1.4 - 3 * x) * np.sin(18 * x)
func, func_grad = function, function_grad

# Data
data = Data([13, 2], [0, 1], "uniform", nums_test=100)

# Neural Network
initializer = nn.initializer.XavierUniform()
net_pinns = FCNet([1] + [20] * 3 + [1], nn_init=initializer)
optimizer = "Adam"

# PINNs model
# The neural network structure net, the pde loss is NNFunc.
func_PINNs = PhysicsInformedNeuralNetwork(net_pinns, NNFunc, func, optimizer)

# Train the model with module Solver
solver_PINNs = Solver(data, func_PINNs, config)

# Train
solver_PINNs.train()


#################### gPINNs ####################
# Pde loss.
def gNNFunc(x, y):
    # dy_x = gradients(y, x)
    # x.stop_gradient = False
    dy_x = paddle.grad(y, x, retain_graph=True, create_graph=True)[0]

    return [
        y + (1.4 - 3 * x) * paddle.sin(18 * x),
        dy_x + 18 * (1.4 - 3 * x) * paddle.cos(18 * x) - 3 * paddle.sin(18 * x),
    ]


# Neural Network
net_gpinns = FCNet([1] + [20] * 3 + [1], nn_init=initializer)

# gPINNs model
# The neural network structure net, the pde loss is NNFunc.
func_gPINNs = PhysicsInformedNeuralNetwork(net_gpinns, gNNFunc, func, optimizer, w_g=1)

# Train the model with module Solver
solver_gPINNs = Solver(data, func_gPINNs, config)

# Train
solver_gPINNs.train()

# Plot.
plt.rcParams.update({"font.size": 16})
#################### Figure 1. C, D ####################
# Figure C
plt.figure()

# true
x0 = np.reshape(np.linspace(0, 1, 1000), (1000, 1))
plt.plot(x0, func(x0), label="Exact", color="black")

x1 = np.reshape(np.linspace(0, 1, 15), (15, 1))
plt.plot(x1, func(x1), color="black", marker="o", linestyle="none")

# predict
# pinns
u_pred_pinns = solver_PINNs.predict(paddle.to_tensor(x0, dtype='float32'))[0].numpy()
plt.plot(x0, u_pred_pinns, label="NN", color="blue", linestyle="dashed")
# gpinns
u_pred_gpinns = solver_gPINNs.predict(paddle.to_tensor(x0, dtype='float32'))[0].numpy()
plt.plot(x0, u_pred_gpinns, label="gNN", color="red", linestyle="dashed")

# label and others params
plt.xlabel("x")
plt.ylabel("u")
plt.legend(frameon=False)
plt.savefig('.\\result\\figure\\function\\u.png', dpi=120)

# Figure D
plt.figure()

# true
x0_g = np.reshape(np.linspace(0, 1, 1000), (1000, 1))
plt.plot(x0_g, func_grad(x0_g), label="Exact", color="black")

x1_g = np.reshape(np.linspace(0, 1, 15), (15, 1))
plt.plot(x1_g, func_grad(np.array(x1_g)), color="black", marker="o", linestyle="none")

u_g_pred_pinns = solver_PINNs.predict(paddle.to_tensor(x0_g, dtype='float32'))[1].numpy()
plt.plot(x0_g, u_g_pred_pinns, label="NN", color="blue", linestyle="dashed")

u_g_pred_gpinns = solver_gPINNs.predict(paddle.to_tensor(x0_g, dtype='float32'))[1].numpy()
plt.plot(x0_g, u_g_pred_gpinns, label="gNN", color="red", linestyle="dashed")

# label and others params
plt.xlabel("x")
plt.ylabel("u`")
plt.legend(frameon=False)
plt.savefig('.\\result\\figure\\function\\u`.png', dpi=120)
# plt.show()

#################### Figure 1. A, B ####################
training_points = np.linspace(5, 30, 26)

l2_error_u = {}
l2_error_u_g = {}
# mean_pde_residual = {}

for training_point in training_points:
    training_point = int(training_point)
    print("#### Start training with training point: {} ####".format(training_point))

    # DATA.
    data_f_ab = Data([training_point - 2, 2], [0, 1], 'random', 100)

    # Training with different training points.
    # Training PINNs.
    solver_PINNs = Solver(data_f_ab, func_PINNs, config)
    solver_PINNs.train()

    # Training gPINNs.
    solver_gPINNs = Solver(data_f_ab, func_gPINNs, config)
    solver_gPINNs.train()

    # Save loss.
    y_true_u, y_true_u_g = function(data_f_ab.test_data), function_grad(data_f_ab.test_data)

    # PINNs.
    y_pred_u_pinns, y_pred_u_g_pinns = solver_PINNs.predict(data_f_ab.test_data)[0], \
                                       solver_PINNs.predict(data_f_ab.test_data)[1]
    l2_u_pinn = solver_PINNs.l2_relative_error(y_true_u, y_pred_u_pinns)
    l2_u_g_pinn = solver_PINNs.l2_relative_error(y_true_u_g, y_pred_u_g_pinns)

    # gPINNs
    y_pred_u_gpinns, y_pred_u_g_gpinns = solver_gPINNs.predict(data_f_ab.test_data), \
                                         solver_gPINNs.predict(data_f_ab.test_data)
    l2_u_gpinn = solver_gPINNs.l2_relative_error(y_true_u, y_pred_u_gpinns)
    l2_u_g_gpinn = solver_gPINNs.l2_relative_error(y_true_u_g, y_pred_u_g_gpinns)

    # Add loss in dict.
    l2_error_u['training_point-{}'.format(training_point)] = [l2_u_pinn, l2_u_gpinn]
    l2_error_u_g['training_point-{}'.format(training_point)] = [l2_u_g_pinn, l2_u_g_gpinn]

# Add loss in different PINNs loss.
l2_pinn_u = []
l2_gpinn_u = []
l2_pinn_u_g = []
l2_gpinn_u_g = []

for i in training_points:
    l2_pinn_u.append(l2_error_u['training_point-{}'.format(int(i))][0])
    l2_gpinn_u.append(l2_error_u['training_point-{}'.format(int(i))][1])
    l2_pinn_u_g.append(l2_error_u_g['training_point-{}'.format(int(i))][0])
    l2_gpinn_u_g.append(l2_error_u_g['training_point-{}'.format(int(i))][1])

# Figure A.
plt.figure(dpi=120)
plt.plot(training_points, l2_pinn_u, 'o-', color='b', label='NN')
plt.plot(training_points, l2_gpinn_u, 's-', color='red', label='gNN')
plt.xlabel('No. of training points')
plt.ylabel(r'L^2'' relative error of u')
plt.legend(frameon=False, loc='best', fontsize=10)
plt.savefig('.\\result\\figure\\function\\figure1_A.png', dpi=120)

# Figure B
plt.figure(dpi=120)
plt.plot(training_points, l2_pinn_u_g, 'o-', color='b', label='NN')
plt.plot(training_points, l2_gpinn_u_g, 's-', color='red', label='gNN')
plt.xlabel('No. of training points')
plt.ylabel(r'L^2'' relative error of u`')
plt.legend(frameon=False, loc='best', fontsize=10)
plt.savefig('.\\result\\figure\\function\\figure1_B.png', dpi=120)
plt.show()
