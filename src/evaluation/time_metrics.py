#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: time_metrics.py
@desc: All metric utils regarding time measures.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
from time import time
from tensorflow.keras.callbacks import Callback


class TimeHistory(Callback):
    """
    Custom keras callback logging duration of each epoch.
    """

    def on_train_begin(self, logs: dict = None):
        """
        Initialize attributes.

        :param logs: Dictionary containing time measures.
        """
        if logs is None:
            logs = {}
        self.on_train_begin_time = time()
        self.times = []
        self.average = []

    def on_epoch_begin(self, batch: int, logs: dict = None):
        """
        Start counting time for epoch.

        :param batch: Number of batch.
        :param logs: Dictionary containing time measures.
        """
        if logs is None:
            logs = {}
        self.epoch_time_start = time()

    def on_epoch_end(self, batch: int, logs: dict = None):
        """
        End counting time for epoch.

        :param batch: Number of epochs.
        :param logs: Dictionary containing time measures.
        """
        if logs is None:
            logs = {}
        self.times.append(time() - self.on_train_begin_time)
        self.average.append(time() - self.epoch_time_start)


def timeit(function):
    """
    Time passed function as a decorator.
    """
    def timed(*args: list, **kwargs: dict):
        """
        Measure time of given function.

        :param args: List of arguments of given function.
        :param kwargs: Dictionary of arguments of given function.
        """
        start = time()
        result = function(*args, **kwargs)
        stop = time()
        return result, stop-start
    return timed
