#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: prepare_data.py
@desc: Load the data, reformat it to have [SAMPLES, ....] dimensions,
split it into train, test and val sets and save them in .h5 file with
'train', 'val' and 'test' groups, each having 'data' and 'labels' keys.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import clize
from clize.parameters import multi

import src.utils.io as io
import src.utils.utils as utils
import src.utils.preprocessing as preprocessing

EXTENSION = 1


def main(*,
         data_file_path: str,
         ground_truth_path: str,
         train_size: ('train_size', multi(min=0)),
         val_size: float = 0.1,
         background_label: int = 0,
         neighborhood_size: int = None,
         channels_idx: int = 0,
         seed: int = 0):
    """
    :param data_file_path: Path to the data file. Supported types are: .npy.
    :param ground_truth_path: Path to the data file.
    :param train_size: If float, should be between 0.0 and 1.0.
        If stratified = True, it represents percentage of each class to be extracted,
        If float and stratified = False, it represents percentage of the whole
        dataset to be extracted with samples drawn randomly, regardless of their class.
        If int and stratified = True, it represents number of samples to be
        drawn from each class.
        If int and stratified = False, it represents overall number of samples
        to be drawn regardless of their class, randomly.
        Defaults to 0.8
    :type train_size: Union[int, float]
    :type train_size: float or int
    :param val_size: Should be between 0.0 and 1.0. Represents the
        percentage of each class from the training set
        to be extracted as a validation set.
        Defaults to 0.1.
    :param background_label: Label indicating the background in GT file.
    :param neighborhood_size: Neighborhood size of the pixel to extract along
        with its spectral bands. Use only if you are training 2D or 3D
        convolutional model.
    :param channels_idx: Index specifying the channels position in the provided
        data.
    :param seed: Seed used for data shuffling.
    :raises TypeError: When provided data or labels file is not supported.
    """
    train_size = utils.parse_train_size(train_size)
    data, labels = io.load_npy(data_file_path, ground_truth_path)

    if neighborhood_size is None:
        data, labels = preprocessing.reshape_cube_to_2d_samples(data, labels, channels_idx)
    else:
        data, labels = preprocessing.reshape_cube_to_3d_samples(
            data, labels, neighborhood_size, background_label, channels_idx)

    data, labels = preprocessing.remove_nan_samples(data, labels)

    train_x, train_y, val_x, val_y, test_x, test_y = \
        preprocessing.train_val_test_split(data, labels, train_size, val_size, seed)

    return utils._build_data_dict(train_x, train_y, val_x, val_y, test_x, test_y)


if __name__ == '__main__':
    clize.run(main)
