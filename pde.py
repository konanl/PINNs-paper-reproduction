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


def poisson_sol(x):
    """ The Analytical Solution of Poisson Eqution"""
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    sol = x + 1 / 8 * np.sin(8 * x)
    for i in range(1, 5):
        sol += 1 / i * np.sin(i * x)
    return sol


def poisson_sol_grad(x):
    """ The Gredient of Poisson Eqution`s Analytical Solution """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    du = 1 + np.cos(8 * x)
    for i in range(1, 5):
        du += np.cos(i * x)
    return du

########################### Diffusion Reaction Equation ###########################


def diffusion_reaction_sol(a):
    """ The Analytic Solution of Diffusion Reaction Equation """
    if not isinstance(a, np.ndarray):
        a = np.array(a)
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


def du_t(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.sin(8 * x_in) / 8
    for i in range(1, 5):
        val += np.sin(i * x_in) / i
    return -np.exp(-t_in) * val


def du_x(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    x_in, t_in = x[:, 0:1], x[:, 1:2]

    val = np.cos(8 * x_in)
    for i in range(1, 5):
        val += np.cos(i * x_in)
    return np.exp(-t_in) * val


########################### Brinkman-Forchheimer Equation ###########################

g = 1
v = 1e-3
K = 1e-3
e = 0.4
H = 1


def BF_sol(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    r = (v*e/(1e-3*K)) ** 0.5
    return g*K/v * (1 - np.cosh(r*(x-H/2))/np.cosh(r*H/2))


