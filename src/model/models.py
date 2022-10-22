#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: models.py
@desc: All models that are used in the project.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import ast
import sys
import tensorflow as tf
from src.model.attention import cbam_block


def _get_model(model_key: str, **kwargs):
    """
    Get a given instance of model specified by model_key.

    :param model_key: Specifies which model to use.
    :param kwargs: Any keyword arguments that the model accepts.
    """
    # Get the list of all model creating functions and their name as the key:
    all_ = {
        str(f): ast.literal_eval(f) for f in dir(sys.modules[__name__])
    }
    return all_[model_key](**kwargs)


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


def pixel_based_fnnc(n_classes: int, input_size: int) -> tf.keras.models.Model:
    """
    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
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


def pixel_based_dacn(n_classes: int, input_size: int) -> tf.keras.models.Model:
    """
    Model for the hyperspectral unmixing which utilizes 
    a pixel-based Dual Attention Convolutional Network.

    :param n_classes: Number of classes.
    :param input_size: Number of input spectral bands.
    :return: Model proposed in the publication listed above.
    """
    input = tf.keras.layers.Input(shape=(input_size, 1))


    Conv1_1 = tf.keras.layers.Conv1D(filters=8, kernel_size=5,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal',
                                     data_format='channels_last')(input)
    LayerN_1 = tf.keras.layers.LayerNormalization(axis=1)(Conv1_1)
    LeakyReLu1_1 = tf.keras.layers.LeakyReLU()(LayerN_1)
    Conv1_2 = tf.keras.layers.Conv1D(filters=16, kernel_size=3,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal')(LeakyReLu1_1)
    LeakyReLu1_2 = tf.keras.layers.LeakyReLU()(Conv1_2)
    Pooling_1 = tf.keras.layers.MaxPool1D(pool_size=2)(LeakyReLu1_2)
    # Convolutional Block Attention Module
    CBAM_1 = cbam_block(Pooling_1)

    #
    Conv2_1 = tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal')(CBAM_1)
    LayerN_2 = tf.keras.layers.LayerNormalization(axis=1)(Conv2_1)
    LeakyReLu2_1 = tf.keras.layers.LeakyReLU()(LayerN_2)
    Conv2_2 = tf.keras.layers.Conv1D(filters=64, kernel_size=3,
                                     padding='same', use_bias=False,
                                     kernel_initializer='he_normal')(LeakyReLu2_1)
    LeakyReLu2_2 = tf.keras.layers.LeakyReLU()(Conv2_2)
    Pooling_2 = tf.keras.layers.MaxPool1D(pool_size=2)(LeakyReLu2_2)
    # Convolutional Block Attention Module
    CBAM_2 = cbam_block(Pooling_2)

    flatten = tf.keras.layers.Flatten()(CBAM_2)
    dense_1 = tf.keras.layers.Dense(units=192, activation='relu')(flatten)
    dense_2 = tf.keras.layers.Dense(units=150, activation='relu')(dense_1)
    dense_3 = tf.keras.layers.Dense(units=n_classes, activation='softmax')(dense_2)

    model = tf.keras.models.Model(inputs=input, outputs=dense_3)
    return model
