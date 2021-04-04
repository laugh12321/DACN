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
from src.model.attention import attach_attention_module


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


def pixel_based_cnn(n_classes: int, input_size: int) -> tf.keras.Sequential:
    """
    Model for pixel-based supervised hyperspectral unmixing proposed in
    the following publication (Chicago style citation):

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
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


def pixel_based_fnn(n_classes: int, input_size: int) -> tf.keras.models.Model:
    """
    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :return: FNN model instance.
    """
    input = tf.keras.layers.Input(shape=(input_size, 1))

    lstm1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=10, return_sequences=True, dropout=0.2))(input)
    lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
    lstm2 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=10, return_sequences=True, dropout=0.3))(lstm1)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    lstm3 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=10, return_sequences=True, dropout=0.4))(lstm2)
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
    Abadunce = tf.keras.layers.Dense(units=n_classes, activation='softmax')(Global) 

    model = tf.keras.models.Model(inputs=input, outputs=Abadunce)
    return model


def pixel_based_bilstm(n_classes: int, input_size: int) -> tf.keras.models.Model:
    """
    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :return: BiLSTM model instance.
    """
    input = tf.keras.layers.Input(shape=(input_size, 1))

    lstm1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=10, return_sequences=True, dropout=0.2))(input)
    lstm1 = tf.keras.layers.BatchNormalization()(lstm1)
    lstm2 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=10, return_sequences=True, dropout=0.3))(lstm1)
    lstm2 = tf.keras.layers.BatchNormalization()(lstm2)
    lstm3 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(units=10, return_sequences=True, dropout=0.4))(lstm2)
    lstm3 = tf.keras.layers.BatchNormalization()(lstm3)
    X2 = tf.keras.layers.Flatten()(lstm3)

    Den1 = tf.keras.layers.Dense(units=600, activation='relu', use_bias=None)(X2)
    Global = tf.keras.layers.Dense(units=150, activation='relu', use_bias=None)(Den1)
    Abadunce = tf.keras.layers.Dense(units=n_classes, activation='softmax')(Global)

    model = tf.keras.models.Model(inputs=input, outputs=Abadunce)
    return model


def rnn_supervised(n_classes: int, input_size: int) -> tf.keras.Sequential:
    """
    Model for the unmixing which utilizes a recurrent neural network (RNN)
    for extracting valuable information from the spectral domain
    in an supervised manner.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
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


# ############################ DACN ############################
def pixel_based_dacn(n_classes: int, input_size: int) -> tf.keras.models.Model:
    """
    Model for the hyperspectral unmixing which utilizes 
    a pixel-based Dual Attention Convolutional Network.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :return: Model proposed in the publication listed above.
    """
    input = tf.keras.layers.Input(shape=(1, input_size, 1))


    Conv1_1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 5),
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal',
                                     data_format='channels_last')(input)
    LayerN_1 = tf.keras.layers.LayerNormalization()(Conv1_1)
    LeakyReLu1_1 = tf.keras.layers.LeakyReLU()(LayerN_1)
    Conv1_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 4),
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal')(LeakyReLu1_1)
    LeakyReLu1_2 = tf.keras.layers.LeakyReLU()(Conv1_2)
    Pooling_1 = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(LeakyReLu1_2)
    # Convolutional Block Attention Module
    CBAM_1 = attach_attention_module(Pooling_1, 'cbam_block')

    #
    Conv2_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 5),
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal')(CBAM_1)
    LayerN_2 = tf.keras.layers.LayerNormalization()(Conv2_1)
    LeakyReLu2_1 = tf.keras.layers.LeakyReLU()(LayerN_2)
    Conv2_2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 4),
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal')(LeakyReLu2_1)
    LeakyReLu2_2 = tf.keras.layers.LeakyReLU()(Conv2_2)
    Pooling_2 = tf.keras.layers.MaxPool2D(pool_size=(1, 2))(LeakyReLu2_2)
    # Convolutional Block Attention Module
    CBAM_2 = attach_attention_module(Pooling_2, 'cbam_block')

    flatten = tf.keras.layers.Flatten()(CBAM_2)
    dense_1 = tf.keras.layers.Dense(units=192, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(units=150, activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(units=n_classes, activation='softmax')(dense_2)

    model = tf.keras.models.Model(inputs=input, outputs=dense_3)
    return model
