#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: models.py
@desc: All models that are used in the project.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import sys
import numpy as np
import tensorflow as tf
from src.model.attention import Channel_attention, Position_attention


def _get_model(model_key: str, **kwargs):
    """
    Get a given instance of model specified by model_key.

    :param model_key: Specifies which model to use.
    :param kwargs: Any keyword arguments that the model accepts.
    """
    # Get the list of all model creating functions and their name as the key:
    all_ = {
        str(f): eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](**kwargs)


def pixel_based_cnn(n_classes: int, input_size: int,
                    **kwargs) -> tf.keras.Sequential:
    """
    Model for pixel-based supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv3D(filters=3, kernel_size=(1, 1, 5),
                               activation='relu',
                               input_shape=(1, 1, input_size, 1),
                               data_format='channels_last'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=12, kernel_size=(1, 1, 5),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Conv3D(filters=24, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def cube_based_cnn(n_classes: int, input_size: int,
                   **kwargs) -> tf.keras.Sequential:
    """
    Model for cube-based supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 5),
                                     activation='relu',
                                     input_shape=(kwargs['neighborhood_size'],
                                                  kwargs['neighborhood_size'],
                                                  input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 5),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 1, 4),
                                     activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def rnn_supervised(n_classes: int, **kwargs) -> tf.keras.Sequential:
    """
    Model for the unmixing which utilizes a recurrent neural network (RNN)
    for extracting valuable information from the spectral domain
    in an supervised manner.

    :param n_classes: Number of classes.
    :param kwargs: Additional arguments.
    :return: RNN model instance.
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.GRU(units=8, input_shape=(kwargs['input_size'], 1),
                            return_sequences=True))
    model.add(tf.keras.layers.GRU(units=32, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=128, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=512, return_sequences=False))

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model


# ############################ attention ############################
def pixel_based_dacn(n_classes: int, input_size: int, **kwargs) -> tf.keras.Model:
    """
    Model for the hyperspectral unmixing which utilizes 
    a pixel-based Dual Attention Convolutional Network.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    input = tf.keras.layers.Input(shape=(1, 1, input_size, 1))

    conv1 = tf.keras.layers.Conv3D(filters=3, kernel_size=(1, 1, 5),
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   data_format='channels_last',
                                   activation='relu')(input)                       
    conv2 = tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 4), 
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   activation='relu')(conv1)

    # POSITION ATTENTION MOUDLE
    pam = Position_attention(ratio=2)(conv2)
    pam = tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 4),
                                 padding='same', use_bias=False,
                                 kernel_initializer='he_normal',
                                 activation='relu')(pam)
    pam = tf.keras.layers.Dropout(0.2)(pam)
    pam = tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 2),
                                 padding='same', use_bias=False,
                                 kernel_initializer='he_normal',
                                 activation='relu')(pam)
    # CHANNEL ATTENTION MOUDLE
    cam = Channel_attention()(conv2)
    cam = tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 4),
                                 padding='same', use_bias=False,
                                 kernel_initializer='he_normal',
                                 activation='relu')(cam)
    cam = tf.keras.layers.Dropout(0.2)(cam)
    cam = tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 2),
                                 padding='same', use_bias=False,
                                 kernel_initializer='he_normal',
                                 activation='relu')(cam)
    # merge
    feature_sum = tf.keras.layers.add([pam, cam])
    feature_sum = tf.keras.layers.Dropout(0.2)(feature_sum)
    feature_sum = tf.keras.layers.Conv3D(filters=6, kernel_size=(1, 1, 2),
                                         padding='same', use_bias=False,
                                         kernel_initializer='he_normal',
                                         activation='relu')(feature_sum)
    merge = tf.keras.layers.concatenate([conv2, feature_sum])

    conv3 = tf.keras.layers.Conv3D(filters=12, kernel_size=(1, 1, 5),
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   activation='relu')(merge)
    pool3 = tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv3D(filters=24, kernel_size=(1, 1, 4),
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   activation='relu')(pool3)
    pool4 = tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2))(conv4)

    flatten = tf.keras.layers.Flatten()(pool4)
    dense_1 = tf.keras.layers.Dense(units=192, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(units=150, activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(units=n_classes, activation='softmax')(dense_2)

    model = tf.keras.Model(inputs=input, outputs=dense_3)
    return model


def cube_based_dacn(n_classes: int, input_size: int,
                             **kwargs) -> tf.keras.Model:
    """
    Model for the hyperspectral unmixing which utilizes 
    a cube-based Dual Attention Convolutional Network.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    input = tf.keras.layers.Input(shape=(kwargs['neighborhood_size'],
                                         kwargs['neighborhood_size'],
                                         input_size, 1))

    conv1 = tf.keras.layers.Conv3D(filters=16, kernel_size=(1, 1, 5),
                                   activation='relu', padding='same',
                                   kernel_initializer='he_normal',
                                   data_format='channels_last')(input)
    conv2 = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 4),
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   activation='relu')(conv1)

    # POSITION ATTENTION MOUDLE
    pam = Position_attention()(conv2)
    pam = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 4),
                                 activation='relu', padding='same',
                                 kernel_initializer='he_normal')(pam)
    pam = tf.keras.layers.Dropout(0.2)(pam)
    pam = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 2),
                                 padding='same', use_bias=False,
                                 kernel_initializer='he_normal',
                                 activation='relu')(pam)
    # CHANNEL ATTENTION MOUDLE
    cam = Channel_attention()(conv2)
    cam = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 4),
                                 padding='same', use_bias=False,
                                 kernel_initializer='he_normal',
                                 activation='relu')(cam)
    cam = tf.keras.layers.Dropout(0.2)(cam)
    cam = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 2),
                                 padding='same', use_bias=False,
                                 kernel_initializer='he_normal',
                                 activation='relu')(cam)
    # merge
    feature_sum = tf.keras.layers.add([pam, cam])
    feature_sum = tf.keras.layers.Dropout(0.2)(feature_sum)
    feature_sum = tf.keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 2),
                                         padding='same', use_bias=False,
                                         kernel_initializer='he_normal',
                                         activation='relu')(feature_sum)
    merge = tf.keras.layers.concatenate([conv2, feature_sum])

    conv3 = tf.keras.layers.Conv3D(filters=64, kernel_size=(1, 1, 5),
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   activation='relu')(merge)
    pool3 = tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2))(conv3)

    conv4 = tf.keras.layers.Conv3D(filters=128, kernel_size=(1, 1, 4),
                                   padding='same', use_bias=False,
                                   kernel_initializer='he_normal',
                                   activation='relu')(pool3)
    pool4 = tf.keras.layers.MaxPool3D(pool_size=(1, 1, 2))(conv4)

    flatten = tf.keras.layers.Flatten()(pool4)
    dense_1 = tf.keras.layers.Dense(units=192, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(units=150, activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(units=n_classes, activation='softmax')(dense_2)

    model = tf.keras.Model(inputs=input, outputs=dense_3)
    return model