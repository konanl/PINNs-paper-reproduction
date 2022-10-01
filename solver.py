#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gPINNs_re 
@File    ：solver.py
@Author  ：LiangL. Yan
@Date    ：2022/9/26 19:44 
"""
import os.path

import paddle
from paddle import nn as nn
import paddle.optimizer as optim
from model import gradients
import numpy as np
import time
import datetime


class Solver(object):
    """Solver for training and testing PINNs & gPINNs."""

    def __init__(self, data, model, config):
        super(Solver, self).__init__()

        # Data loader.
        self.data = data

        # Basic physics informed neural network.
        self.model = model

        # Model configurations.
        self.optimizer = None

        # Training configurations.
        self.lr = config.lr
        self.num_epochs = config.num_epochs
        self.resume_epochs = config.resume_epochs
        # self.training_points = config.training_points

        # Test configurations.
        # how much step we should test
        self.test_epochs = config.test_epochs

        # Directories.
        self.model_save_dir = config.model_save_dir
        self.result_dir = config.result_dir

        # Step size.
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        # Initialize model
        self.build_model()

        # PDE loss
        # self.pde_loss = None

    def build_model(self):
        """Create PINNs or gPINNs."""
        # 1. Chose optimizer
        if self.model.optimizer == "Adam":
            self.optimizer = optim.Adam(parameters=self.model.net.parameters(), learning_rate=self.lr)
        elif self.model.optimizer == "Adam+L-BFGS":
            pass
        # 2. Build/Construct loss function ★
        if self.whatPINNs():
            print("Start build PINNs model...")
            self.print_network(model=self.model.net, name="PINNs")
            print("Success build PINNs model.")
        else:
            print("Start build gPINNs with w_g = {}".format(self.model.w_g))
            self.print_network(model=self.model.net, name="gPINNs with w_g = {}".format(self.model.w_g))
            print("Success build gPINNs with w_g = {}".format(self.model.w_g))

    def whatPINNs(self):
        """Judge what type of PINNs it is."""
        if self.model.w_g != 0:
            # gPINNs
            return False
        # PINNs
        return True

    @staticmethod
    def print_network(model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def restore_model(self, resume_epochs):
        """Restore the trained PINNs or gPINNs."""
        print('Loading the trained models from step {}...'.format(resume_epochs))
        if not self.whatPINNs():
            path = os.path.join(
                self.model_save_dir,
                '{}-gPINNs-w_g-{}.pdparams'.format(resume_epochs, str(self.model.w_g)))
        else:
            path = os.path.join(self.model_save_dir, '{}-PINNs.pdparams'.format(resume_epochs))
        self.model.net.set_state_dict(paddle.load(path))
        print("Success load model!")

    def loss(self, x, y):
        """Loss function of PINNs or gPINNs."""
        loss = self.model.PDE(x, y)
        loss_f = loss[0]
        if not self.whatPINNs():
            if len(loss) == 2:
                loss_g = loss[1]
                return \
                    self.model.w_f * paddle.mean(paddle.square(loss_f)) \
                    + self.model.w_g * paddle.mean(paddle.square(loss_g))
            elif len(loss) > 2:
                loss_g = 0
                for loss_ in loss[1:]:
                    loss_g += self.model.w_g * loss_
                return \
                    self.model.w_f * paddle.mean(paddle.square(loss_f)) \
                    + self.model.w_g * paddle.mean(paddle.square(loss_g))
        return paddle.mean(paddle.square(loss_f))

    @staticmethod
    def mean_square_error(y_true, y_pred):
        """Mean Square Error."""
        if isinstance(y_true, paddle.Tensor) or isinstance(y_pred, paddle.Tensor):
            y_true, y_pred = paddle.to_tensor(y_true), paddle.to_tensor(y_pred)
        return paddle.mean(paddle.square(y_true - y_pred))

    @staticmethod
    def l2_relative_error(y_true, y_pred):
        """L2 norm relative error."""
        if isinstance(y_pred, np.ndarray):
            return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)
        else:
            y_pred = np.array(y_pred)
            return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true)

    @staticmethod
    def mean_l2_relative_error(y_true, y_pred):
        """Compute the average of L2 relative error along the first axis."""
        return np.mean(
            np.linalg.norm(y_true - y_pred, axis=1) / np.linalg.norm(y_true, axis=1)
        )

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.optimizer.clear_grad()

    def train(self):
        """Train PINNs/gPINNs"""

        # =================================================================================== #
        #                             1. Preprocess input data                                #
        # =================================================================================== #

        # Process train data.
        if isinstance(self.data.train_data, np.ndarray):
            train_data = paddle.to_tensor(self.data.train_data)
        elif isinstance(self.data.train_data, paddle.Tensor):
            train_data = self.data.train_data

        # Learning rate
        lr = self.lr

        # Start training from scratch or resume training.
        start_epochs = 0
        if self.resume_epochs:
            start_epochs = self.resume_epochs
            self.restore_model(self.resume_epochs)

        # Start train.
        if self.whatPINNs():
            print('Start training PINNs...')
        else:
            print('Start training gPINNs w_g = {}'.format(str(self.model.w_g)))
        start_time = time.time()
        for epoch in range(start_epochs, self.num_epochs):

            # =================================================================================== #
            #                             2. Train the PINNs/gPINNs                               #
            # =================================================================================== #

            # Compute the loss
            train_data.stop_gradient = False
            y_pred = self.model(train_data)
            train_loss = self.loss(train_data, y_pred)
            # loss_pde = self.mean_square_error(0, self.model(train_data))
            # loss_u, loss_u`

            # Backward and optimize.
            train_loss.backward()
            self.optimizer.minimize(train_loss)
            self.optimizer.step()
            self.reset_grad()

            # Logging.
            loss = {}
            if self.whatPINNs():
                loss['PINNs/loss'] = train_loss.item()
            else:
                loss['gPINNs, w={}/loss'.format(self.model.w_g)] = train_loss.item()

            # =================================================================================== #
            #                               3. Miscellaneous                                      #
            # =================================================================================== #

            # Print out training information.
            if (epoch+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, epoch+1, self.num_epochs)
                for tag, value in loss.items():
                    log += ", {}: {:.2e}".format(tag, value)
                print(log)

            # # Print out testing information
            # if (epoch+1) % self.test_epochs == 0:
            #     test_loss, metric = self.test()
            #     print("#### Model [{}], W_g [{}], Test loss [{}] Test loss [{:.4f}], Test metric [{:.4f}] ####".format(
            #         'PINNs' if self.whatPINNs() else 'gPINNs', self.model.w_g, test_loss, metric
            #     ))

            # Save model checkpoints.
            if (epoch+1) % self.model_save_step == 0:
                if self.whatPINNs():
                    path = os.path.join(self.model_save_dir, '{}-PINNs.pdparams'.format(epoch+1))
                else:
                    path = os.path.join(self.model_save_dir,
                                        '{}-gPINNs-w_g-{}.pdparams'.format(epoch+1, self.model.w_g))
                paddle.save(self.model.net.state_dict(), path)
                print('Saved model checkpoints into {}...'.format(self.model_save_dir))

    def test(self):
        """Test the trained model and print it`s metric."""
        # Load the trained PINNs/gPINNs model.
        self.restore_model(self.test_epochs)

        # Set data loader
        if isinstance(self.data.test_data, np.ndarray):
            test_data = paddle.to_tensor(self.data.test_data)
        else:
            test_data = self.data.test_data

        self.model.net.eval()
        y_pred = self.model(test_data)
        y_true = self.model.solution(test_data)
        test_loss = self.loss(test_data, y_pred)
        metrics = self.l2_relative_error(y_true, y_pred)
        return test_loss, metrics

    def predict(self, X):
        """Predict with trained model."""
        if not isinstance(X, paddle.Tensor):
            X = paddle.to_tensor(X, dtype='float32')
        # Loading trained model
        self.model.net.eval()
        X.stop_gradient = False
        self.restore_model(self.num_epochs)
        y_pred = self.model(X)
        dy_x_pred = paddle.grad(y_pred, X, retain_graph=False, create_graph=False)[0]
        return [y_pred, dy_x_pred]
