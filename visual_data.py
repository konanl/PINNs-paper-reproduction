#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：gPINNs_re 
@File    ：visual_data.py
@Author  ：LiangL. Yan
@Date    ：2022/9/26 21:26 
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import *
import os

""" Plot the result."""


class Plot(object):

    def __init__(self):
        """ what? """

        # self.save_dir = save_dir
        self.font = {"size": 16, 'family': 'serif'}

    def plot_loss(self, x, y, label, title=None):
        """ Plot the train/test loss and save the output """
        pass

    def plot_predict(self, x, y, label, ylabel, color, title=None, marker=None, linestyle="dashed"):
        """ Plot the output of predict """
        plt.plot(x, y, label=label, color=color, marker=marker, linestyle=linestyle)
        plt.xlabel("x", self.font)
        plt.ylabel(ylabel, self.font)
        plt.yticks(fontproperties=self.font['family'], size=self.font["size"])
        plt.xticks(fontproperties=self.font['family'], size=self.font["size"])
        plt.title(title, self.font)
        plt.legend(frameon=False)

    def plot_l2(self):
        pass