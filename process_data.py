#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gPINNs_re 
@File    ：process_data.py
@Author  ：LiangL. Yan
@Date    ：2022/9/26 11:03 
"""

import numpy as np
import random
import skopt

###############################
## Generate dataset and load ##
###############################

LARGE_INT = 1000000


def data_generate(nums, border, left):
    """Generate random data."""
    data_list = []
    for i in range(nums):
        data_random = random.randint(-LARGE_INT * left, LARGE_INT) * 1.0 / LARGE_INT
        if data_random not in data_list:
            data_random *= border
            data_list.append(data_random)

    return data_list


class Data(object):
    """The class for generate train data and test data."""

    def __init__(self, nums, scope, train_distribution, nums_test=None):
        """
        Initialize data class
        nums:       the size of data --[num.domain, mum.boundary]
        scope:      the border of data --[left, right]
            scope.l:        left of scope
            scope.r:        right of scope
        train_distribution:     the train distribution: 'uniform', 'random'
        nums_test:      the size of test data
        train_data:     the training dataset
        test_data:      the testing dataset
        """
        self.num_domain, self.num_boundary = nums[0], nums[1]
        self.l, self.r = scope[0], scope[1]
        self.train_distribution = train_distribution
        self.nums_test = nums_test
        self.train_data = None
        self.test_data = None

        # Generate data
        self.gen_dataset()

    def uniform_points(self, n, boundary=True):
        """Generate uniform distribution dataset."""
        if boundary:
            return np.linspace(self.l, self.r, num=n, dtype='float32')[:, None]
        return np.linspace(self.l, self.r, num=n + 1, endpoint=False, dtype='float32')[1:None]

    def uniform_boundary_points(self, n):
        """Generate uniform distribution of the boundary."""
        if n == 1:
            return np.array([[self.l]]).astype('float32')
        xl = np.full((n // 2, 1), self.l).astype('float32')
        xr = np.full((n - n // 2, 1), self.r).astype('float32')
        return np.vstack((xl, xr))

    def random_points(self, n, random="pseudo"):
        x = sample(n, 1, random)
        return x * (self.r - self.l) + self.l

    def random_boundary_points(self, n, random="pseudo"):
        if n == 2:
            return np.array([[self.l], [self.r]]).astype('float32')
        return np.random.choice([self.l, self.r], n).astype('float32')

    def gen_dataset(self, boundary=True):
        """ Generate dataset with class Data."""
        # train data
        if self.train_distribution == 'uniform':
            if not boundary:
                self.train_data = np.vstack((
                    self.uniform_points(self.num_domain, boundary=False).reshape(self.num_domain, -1),
                    self.uniform_boundary_points(self.num_boundary))
                )
            self.train_data = np.vstack((self.uniform_points(self.num_domain),
                                         self.uniform_boundary_points(self.num_boundary)))
        elif self.train_distribution == 'random':
            if self.num_boundary > 2:
                self.train_data = np.vstack((
                    self.random_points(self.num_domain),
                    self.random_boundary_points(self.num_boundary).reshape(self.num_boundary, -1))
                )
            self.train_data = np.vstack((
                self.random_points(self.num_domain),
                self.random_boundary_points(self.num_boundary))
            )
        # test data
        self.test_data = self.random_points(self.nums_test)


# def gen_dataset(data, boundary=True):
#     """ Generate dataset with class Data."""
#     # train data
#     if data.train_distribution == 'uniform':
#         if not boundary:
#             data.train_data = np.vstack((
#                 data.uniform_points(data.num_domain, boundary=False).reshape(data.num_domain, -1),
#                 data.uniform_boundary_points(data.num_boundary))
#             )
#         data.train_data = np.vstack((data.uniform_points(data.num_domain),
#                                      data.uniform_boundary_points(data.num_boundary)))
#     elif data.train_distribution == 'random':
#         if data.num_boundary > 2:
#             data.train_data = np.vstack((
#                 data.random_points(data.num_domain),
#                 data.random_boundary_points(data.num_boundary).reshape(data.num_boundary, -1))
#             )
#         data.train_data = np.vstack((
#                 data.random_points(data.num_domain),
#                 data.random_boundary_points(data.num_boundary))
#         )
#     # test data
#     data.test_data = data.random_points(data.nums_test)


#################### Random Sample ####################
# from DeepXDE
def sample(n_samples, dimension, sampler="pseudo"):
    """Generate pseudorandom or quasirandom samples in [0, 1]^dimension.

    Args:
        n_samples (int): The number of samples.
        dimension (int): Space dimension.
        sampler (string): One of the following: "pseudo" (pseudorandom), "LHS" (Latin
            hypercube sampling), "Halton" (Halton sequence), "Hammersley" (Hammersley
            sequence), or "Sobol" (Sobol sequence).
    """
    if sampler == "pseudo":
        return pseudorandom(n_samples, dimension)
    if sampler in ["LHS", "Halton", "Hammersley", "Sobol"]:
        return quasirandom(n_samples, dimension, sampler)
    raise ValueError("f{sampler} sampling is not available.")


def pseudorandom(n_samples, dimension):
    """Pseudo random."""
    # If random seed is set, then the rng based code always returns the same random
    # number, which may not be what we expect.
    # rng = np.random.default_rng(config.random_seed)
    # return rng.random(size=(n_samples, dimension), dtype=config.real(np))
    return np.random.random(size=(n_samples, dimension)).astype('float32')


def quasirandom(n_samples, dimension, sampler):
    # Certain points should be removed:
    # - Boundary points such as [..., 0, ...]
    # - Special points [0, 0, 0, ...] and [0.5, 0.5, 0.5, ...], which cause error in
    #   Hypersphere.random_points() and Hypersphere.random_boundary_points()
    skip = 0
    if sampler == "LHS":
        sampler = skopt.sampler.Lhs()
    elif sampler == "Halton":
        # 1st point: [0, 0, ...]
        sampler = skopt.sampler.Halton(min_skip=1, max_skip=1)
    elif sampler == "Hammersley":
        # 1st point: [0, 0, ...]
        if dimension == 1:
            sampler = skopt.sampler.Hammersly(min_skip=1, max_skip=1)
        else:
            sampler = skopt.sampler.Hammersly()
            skip = 1
    elif sampler == "Sobol":
        # 1st point: [0, 0, ...], 2nd point: [0.5, 0.5, ...]
        sampler = skopt.sampler.Sobol(randomize=False)
        if dimension < 3:
            skip = 1
        else:
            skip = 2
    space = [(0.0, 1.0)] * dimension
    return np.asarray(
        sampler.generate(space, n_samples + skip)[skip:], dtype='float32'
    )


if __name__ == '__main__':
    data = Data([13, 2], [0, 1], 'uniform', nums_test=100)
    print(data.train_data.shape)
    print(data.test_data.shape)
