import paddle
import numpy as np

''' The PDE '''

########################### Function ###########################

''' u(x) = -(1.4 - 3 * x) * sin(18 * x) , x->[0,1] '''


def function(x):
    return -(1.4 - 3 * x) * np.sin(18 * x)


def function_grad(x):
    return 3 * np.sin(18 * x) + 18 * (3 * x - 1.4) * np.cos(18 * x)

########################### Poisson Equation ###########################


'''
The Partial Differential Function of Poissin is:
    
    -\Delta u = \sum_{i=1}^{4} i\sin ix + 8\sin 8x, x\in [0, \pi ]
    
'''


def poisson_sol(x):
    """ The Analytical Solution of Poisson Eqution"""
    sol = x + 1 / 8 * paddle.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * paddle.sin(i * x)
    return sol


def poisson_sol_grad(x):
    """ The Gredient of Poisson Eqution`s Analytical Solution """
    du = 1 + paddle.cos(8 * x)
    for i in range(1, 5):
        du += paddle.cos(i * x)
    return du

########################### Diffusion Reaction Equation ###########################
