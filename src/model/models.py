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
from src.model.attention import cbam_block, \
    spatial_attention, channel_attention


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


def rnn_supervised(n_classes: int, input_size: int, 
                   **kwargs) -> tf.keras.Sequential:
    """
    Model for the unmixing which utilizes a recurrent neural network (RNN)
    for extracting valuable information from the spectral domain
    in an supervised manner.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: RNN model instance.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(units=8, input_shape=(input_size, 1),
                                  return_sequences=True))
    model.add(tf.keras.layers.GRU(units=32, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=128, return_sequences=True))
    model.add(tf.keras.layers.GRU(units=512, return_sequences=False))

    model.add(tf.keras.layers.Dense(n_classes, activation='softmax'))
    return model


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
    model.add(tf.keras.layers.Conv1D(filters=3, kernel_size=5,
                                     activation='relu',
                                     input_shape=(input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=6, kernel_size=4,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=12, kernel_size=5,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=24, kernel_size=4,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def pixel_based_dcae(n_classes: int, input_size: int,
                     **kwargs) -> tf.keras.Sequential:
    """
    Model for pixel-based unsupervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):
    
    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.
    
    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv1D(filters=2, kernel_size=3,
                                     activation='relu',
                                     input_shape=(input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2)
    model.add(tf.keras.layers.Conv1D(filters=4, kernel_size=3,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=3,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3,
                                     activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='relu'))
    model.add(tf.keras.layers.Softmax())
    # Decoder part (later to be dropped):
    model.add(tf.keras.layers.Dense(units=input_size, activation='relu'))
    # Set the endmembers weights to be equal to the endmembers matrix i.e.,
    # the spectral signatures of each class:
    model.layers[-1].set_weights(
        (np.swapaxes(kwargs['endmembers'], 1, 0), np.zeros(input_size)))
    # Freeze the last layer which must be equal to endmembers
    # and residual term (zero vector):
    model.layers[-1].trainable = False
    return model


def pixel_based_fnnc(n_classes: int, input_size: int, 
                     **kwargs) -> tf.keras.models.Model:
    """
    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: FNNC model instance.
    """
    input = tf.keras.layers.Input(shape=(input_size, 1))

    lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10, return_sequences=True, 
                                                               dropout=0.2))(input)
    lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
    lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10, 
                                                               return_sequences=True, dropout=0.3))(lstm1)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    lstm3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=10, 
                                                               return_sequences=True, dropout=0.4))(lstm2)
    lstm3 = tf.keras.layers.BatchNormalization()(lstm3)
    X2 = tf.keras.layers.Flatten()(lstm3)

    Global = tf.keras.layers.Conv1D(filters=3, kernel_size=5, activation='relu', padding='same')(input)
    Global = tf.keras.layers.MaxPooling1D(pool_size=2)(Global)

    Global = tf.keras.layers.Conv1D(filters=6, kernel_size=4, activation='relu')(Global)
    Global = tf.keras.layers.MaxPooling1D(pool_size=2)(Global)

    Global = tf.keras.layers.Conv1D(filters=12, kernel_size=5, activation='relu')(Global)
    Global = tf.keras.layers.MaxPooling1D(pool_size=2)(Global)

    Global = tf.keras.layers.Conv1D(filters=24, kernel_size=4, activation='relu')(Global)
    Global = tf.keras.layers.MaxPooling1D(pool_size=2)(Global)

    Global = tf.keras.layers.Flatten()(Global)
    Con = tf.keras.layers.concatenate([X2, Global])

    Den1 = tf.keras.layers.Dense(units=600, activation='relu', use_bias=None)(Con)
    Global = tf.keras.layers.Dense(units=150, activation='relu', use_bias=None)(Den1)
    
    Abadunce = tf.keras.layers.Reshape((1, 150))(Global)
    Abadunce = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=100, dropout=0.2))(Abadunce)
    Abadunce = tf.keras.layers.BatchNormalization()(Abadunce)
    Abadunce = tf.keras.layers.RepeatVector(n_classes)(Abadunce)
    Abadunce = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=50, return_sequences=True, 
                                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4), 
                                                                  dropout=0.2))(Abadunce)
    Abadunce = tf.keras.layers.BatchNormalization()(Abadunce)
    Abadunce = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=20, return_sequences=True, 
                                                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4)))(Abadunce)
    Abadunce = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='sigmoid'))(Abadunce)
    Abadunce = tf.keras.backend.squeeze(Abadunce, axis=-1)
    
    model = tf.keras.models.Model(inputs=input, outputs=Abadunce)
    return model


def pixel_based_dacn(n_classes: int, input_size: int, 
                     **kwargs) -> tf.keras.Sequential:
    """
    Model for the hyperspectral unmixing which utilizes 
    a pixel-based Dual Attention Convolutional Network.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()

    # Dual Attention Convolutional Block
    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal',
                                     input_shape=(input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    # Convolutional Block Attention Module
    model.add(channel_attention())
    model.add(spatial_attention())

    # Dual Attention Convolutional Block
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal'))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    # Convolutional Block Attention Module
    model.add(channel_attention())
    model.add(spatial_attention())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=192, activation='relu'))
    model.add(tf.keras.layers.Dense(units=150, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))
    return model


def pixel_based_dacae(n_classes: int, input_size: int, 
                      **kwargs) -> tf.keras.Sequential:
    """
    Model for the hyperspectral unmixing which utilizes 
    a pixel-based Dual Attention Convolutional Autoencoders.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :param kwargs: Additional arguments.
    :return: Model proposed in the publication listed above.
    """
    model = tf.keras.Sequential()

    # Dual Attention Convolutional Block
    model.add(tf.keras.layers.Conv1D(filters=8, kernel_size=5,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal',
                                     input_shape=(input_size, 1),
                                     data_format='channels_last'))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    # Convolutional Block Attention Module
    model.add(channel_attention())
    model.add(spatial_attention())

    # Dual Attention Convolutional Block
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal'))
    model.add(tf.keras.layers.LayerNormalization(axis=1))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.MaxPool1D(pool_size=2))
    # Convolutional Block Attention Module
    model.add(channel_attention())
    model.add(spatial_attention())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=n_classes, activation='relu'))
    model.add(tf.keras.layers.Softmax())
    # Decoder part (later to be dropped):
    model.add(tf.keras.layers.Dense(units=input_size, activation='relu'))
    # Set the endmembers weights to be equal to the endmembers matrix i.e.,
    # the spectral signatures of each class:
    model.layers[-1].set_weights(
        (np.swapaxes(kwargs['endmembers'], 1, 0), np.zeros(input_size)))
    # Freeze the last layer which must be equal to endmembers
    # and residual term (zero vector):
    model.layers[-1].trainable = False
    return model
