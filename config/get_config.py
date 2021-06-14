#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 1月 30, 2021 

@file: get_config.py
@desc: get parameters
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspectral Unmixing")
    parser.add_argument('--path', nargs='?', default='./datasets/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='urban',
                        help='Choose a dataset.')
    parser.add_argument('--save_path', nargs='?', default='./results',
                        help='Result save path.')
    parser.add_argument('--train_size', type=float or int, default=0.8,
                        help='Training size.')
    parser.add_argument('--test_size', type=int, default=None,
                        help='Number of pixels to subsample the test set .')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Represents the percentage of samples from the training set '
                             'to be extracted as a validation set.')
    parser.add_argument('--model_names', nargs='+', type=str, default='unmixing_cube_based_dcae',
                        help='Name of the model, it serves as a key in the dictionary '
                             'holding all functions returning models.')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbosity mode used in training, (0, 1 or 2).')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Size of the batch used in training phase, '
                             'it is the size of samples per gradient step.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs for model to train.')
    parser.add_argument('--runs', type=int, default=5,
                        help='Number of total experiment runs.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs without improvement in order to '
                             'stop the training phase.')
    return parser.parse_args()


def get_config(filename):
    args = parse_args()
    with open(filename, 'r') as f:
        args.__dict__ = json.load(f)
    return args
