#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1月 21, 2021 

@file: load_data.py
@desc: 
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""

import numpy as np
import scipy.io as sio


class Data(object):

    def __init__(self, path, dataset):
        self.data, self.data_gt = self.load_mat(path, dataset)

    @staticmethod
    def load_mat(path, dateset):
        """

        :type dateset: object
        :param path:
        :param dateset:
        :return:
        """
        data = sio.loadmat(path + dateset + '.mat')
        data_gt = sio.loadmat(path + dateset + '_gt.mat')

        data = np.asarray_chkfinite(data[dateset])
        data_gt = np.asarray_chkfinite(data_gt[dateset + '_gt'])

        return data, data_gt
