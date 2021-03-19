#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: preprocessing.py
@desc: All function related to the data preprocessing step of the pipeline.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import functools
import numpy as np
from itertools import product
from typing import Tuple, Union, List

from src.utils.utils import shuffle_arrays_together, get_label_indices_per_class

HEIGHT = 0
WIDTH = 1
DEPTH = 2


def reshape_cube_to_2d_samples(data: np.ndarray,
                               labels: np.ndarray,
                               channels_idx: int = 0) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Reshape the data and labels from [CHANNELS, HEIGHT, WIDTH] to
        [PIXEL,CHANNELS, 1], so it fits the 2D Convolutional modules.

    :param data: Data to reshape.
    :param labels: Corresponding labels.
    :param channels_idx: Index at which the channels are located in the
                         provided data file.
    :return: Reshape data and labels
    :rtype: tuple with reshaped data and labels
    """
    data = np.rollaxis(data, channels_idx, len(data.shape))
    height, width, channels = data.shape
    data = data.reshape(height * width, channels)
    data = np.expand_dims(np.expand_dims(data, 1), 1)
    labels = labels.reshape(height * width, -1)
    data = data.astype(np.float32)
    labels = labels.astype(np.float32)

    return data, labels


def get_padded_cube(data: np.ndarray, padding_size: int) -> np.ndarray:
    """
    Pad the cube with zeros

    :param data: Data to pad
    :param padding_size: Size of the padding for each side
    :return: Padded cube
    """
    v_padding = np.zeros((padding_size, data.shape[WIDTH], data.shape[DEPTH]))
    data = np.vstack((v_padding, data))
    data = np.vstack((data, v_padding))
    h_padding = np.zeros((data.shape[HEIGHT], padding_size, data.shape[DEPTH]))
    data = np.hstack((h_padding, data))
    data = np.hstack((data, h_padding))
    return data


def reshape_cube_to_3d_samples(data: np.ndarray,
                               labels: np.ndarray,
                               neighborhood_size: int = 5,
                               channels_idx: int = 0) -> \
        Tuple[np.ndarray, np.ndarray]:
    """
    Reshape data to a array of dimensionality:
    [N_SAMPLES, HEIGHT, WIDTH, CHANNELS] and the labels
    to dimensionality of: [N_SAMPLES, N_CLASSES]

    :param data: Data passed as array.
    :param labels: Labels passed as array.
    :param neighborhood_size: Length of the spatial patch.
    :param channels_idx: Index of the channels.
    :rtype: Tuple of data and labels reshaped to 3D format.
    """
    data = np.rollaxis(data, channels_idx, len(data.shape))
    height, width, _ = data.shape
    padding_size = int(neighborhood_size % np.ceil(float(neighborhood_size) / 2.))
    data = get_padded_cube(data, padding_size)
    samples = []
    labels_3d = []
    data = data.astype(np.float32)
    for x, y in product(list(range(height)), list(range(width))):
        samples.append(data[x:x + padding_size * 2 + 1,
                       y:y + padding_size * 2 + 1])
        labels_3d.append(labels[x, y])
    samples = np.array(samples).astype(np.float32)
    labels3d = np.array(labels_3d).astype(np.float32)

    return samples, labels3d


def remove_nan_samples(data: np.ndarray, labels: np.ndarray) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Remove samples which contain only nan values

    :param data: Data with dimensions [SAMPLES, ...]
    :param labels: Corresponding labels
    :return: Data and labels with removed samples containing nans
    """
    all_but_samples_axes = tuple(range(1, data.ndim))
    nan_samples_indexes = np.isnan(data).any(axis=all_but_samples_axes).ravel()
    labels = labels[~nan_samples_indexes]
    data = data[~nan_samples_indexes]
    return data, labels


def train_val_test_split(data: np.ndarray, labels: np.ndarray,
                         train_size: Union[List, float, int] = 0.8,
                         val_size: float = 0.1,
                         seed: int = 0) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into train, val and test sets. The size of the training set
    is set by the train_size parameter. All the remaining samples will be
    treated as a test set

    :param data: Data with the [SAMPLES, ...] dimensions
    :param labels: Vector with corresponding labels
    :param train_size:  If float, should be between 0.0 and 1.0,
        If stratified = True, it represents percentage of each class to be extracted.
        If float and stratified = False, it represents percentage of the whole datasets to be extracted with samples drawn randomly, regardless of their class.
        If int and stratified = True, it represents number of samples to be drawn from each class.
        If int and stratified = False, it represents overall number of samples to be drawn regardless of their class, randomly.
        Defaults to 0.8
    :param val_size: Should be between 0.0 and 1.0. Represents the percentage
                     of each class from the training set to be extracted as a
                     validation set, defaults to 0.1
    :param seed: Seed used for data shuffling
    :return: train_x, train_y, val_x, val_y, test_x, test_y
    :raises AssertionError: When wrong type is passed as train_size
    """
    shuffle_arrays_together([data, labels], seed=seed)
    train_indices = _get_set_indices(train_size, labels)
    val_indices = _get_set_indices(val_size, labels[train_indices])
    val_indices = train_indices[val_indices]
    test_indices = np.setdiff1d(np.arange(len(data)), train_indices)
    train_indices = np.setdiff1d(train_indices, val_indices)
    return data[train_indices], labels[train_indices], data[val_indices], \
           labels[val_indices], data[test_indices], labels[test_indices]


@functools.singledispatch
def _get_set_indices(size: Union[List, float, int],
                     labels: np.ndarray) -> np.ndarray:
    """
    Extract indices of a subset of specified data according to size and
    stratified parameters.

    :param labels: Vector with corresponding labels
    :param size: If float, should be between 0.0 and 1.0,
                    if stratified = True, it represents percentage
                    of each class to be extracted.
                 If float and stratified = False, it represents
                    percentage of the whole datasets to be extracted
                    with samples drawn randomly, regardless of their class.
                 If int and stratified = True, it represents number of samples
                    to be drawn from each class.
                 If int and stratified = False, it represents overall number of
                    samples to be drawn regardless of their class, randomly.
                 Defaults to 0.8
    :return: Indexes of the train set
    :raises TypeError: When wrong type is passed as size
    """
    raise NotImplementedError()


@_get_set_indices.register(float)
def _(size: float, labels: np.ndarray) -> np.ndarray:
    assert 0 < size <= 1
    train_indices = np.arange(int(len(labels) * size))

    return train_indices


@_get_set_indices.register(int)
def _(size: int, labels: np.ndarray) -> np.ndarray:
    assert size >= 1
    train_indices = np.arange(size, dtype=int)

    return train_indices


@_get_set_indices.register(list)
def _(size: List,
      labels: np.ndarray,
      _) -> np.ndarray:
    label_indices, unique_labels = get_label_indices_per_class(
        labels, return_uniques=True)
    if len(size) == 1:
        size = int(size[0])
    for n_samples, label in zip(size, range(len(unique_labels))):
        label_indices[label] = label_indices[label][:int(n_samples)]
    train_indices = np.concatenate(label_indices, axis=0)
    return train_indices
