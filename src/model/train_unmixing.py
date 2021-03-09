#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: train_unmixing.py
@desc: Perform the training of the models for the unmixing problem.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import os
import numpy as np
import tensorflow as tf
from typing import Dict

import src.model.enums as enums
from src.utils import io, transforms
from src.model.models import _get_model
from src.utils.transforms import UNMIXING_TRANSFORMS
from src.evaluation import time_metrics
from src.evaluation.performance_metrics import UNMIXING_LOSSES, \
    UNMIXING_TRAIN_METRICS


def train(data: Dict[str, np.ndarray],
          model_name: str,
          dest_path: str,
          sample_size: int,
          n_classes: int,
          neighborhood_size: int,
          lr: float,
          batch_size: int,
          epochs: int,
          verbose: int,
          shuffle: bool,
          patience: int,
          seed: int):
    """
    Function for running experiments on various unmixing models,
    given a set of hyper parameters.

    :param data: The data dictionary containing
        the subsets for training and validation.
        First dimension of the datasets should be the number of samples.
    :param model_name: Name of the model, it serves as a key in the
        dictionary holding all functions returning models.
    :param dest_path: Path to where all experiment runs will be saved as
        subdirectories in this given directory.
    :param sample_size: Size of the input sample.
    :param n_classes: Number of classes.
    :param neighborhood_size: Size of the spatial patch.
    :param lr: Learning rate for the model, i.e., regulates
        the size of the step in the gradient descent process.
    :param batch_size: Size of the batch used in training phase,
        it is the size of samples per gradient step.
    :param epochs: Number of epochs for model to train.
    :param verbose: Verbosity mode used in training, (0, 1 or 2).
    :param shuffle: Boolean indicating whether to shuffle datasets.
    :param patience: Number of epochs without improvement in order to
        stop the training phase.
    :param seed: Seed for training reproducibility.
    """
    # Reproducibility:
    np.random.seed(seed=seed)

    model = _get_model(
        model_key=model_name,
        **{'input_size': sample_size,
           'n_classes': n_classes,
           'neighborhood_size': neighborhood_size})
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        loss=UNMIXING_LOSSES[model_name],
        metrics=UNMIXING_TRAIN_METRICS[model_name])

    time_history = time_metrics.TimeHistory()

    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        dest_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min')

    callbacks = [time_history, mcp_save, early_stopping]

    train_dict = data[enums.Dataset.TRAIN]
    val_dict = data[enums.Dataset.VAL]

    min_, max_ = data[enums.DataStats.MIN], data[enums.DataStats.MAX]

    transformations = [transforms.MinMaxNormalize(min_=min_, max_=max_)]
    transformations += [t(**{'neighborhood_size': neighborhood_size}) for t
                        in UNMIXING_TRANSFORMS[model_name]]

    train_dict = transforms.apply_transformations(train_dict, transformations)
    val_dict = transforms.apply_transformations(val_dict, transformations)

    history = model.fit(
        x=train_dict[enums.Dataset.DATA],
        y=train_dict[enums.Dataset.LABELS],
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=(val_dict[enums.Dataset.DATA],
                         val_dict[enums.Dataset.LABELS]),
        callbacks=callbacks,
        batch_size=batch_size)

    np.savetxt(os.path.join(dest_path,
                            'min-max.csv'), np.array([min_, max_]),
               delimiter=',', fmt='%f')

    history.history[time_metrics.TimeHistory.__name__] = time_history.average

    io.save_metrics(dest_path=dest_path,
                    file_name='training_metrics.csv',
                    metrics=history.history)
