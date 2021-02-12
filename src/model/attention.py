#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2月 03, 2021 

@file: attention.py
@desc: channel attention layer
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.config.run_functions_eagerly(True)


class Channel_attention(tf.keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super(Channel_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(name='beta',
                                    shape=(self.units, ),
                                    initializer='zeros',
                                    trainable=True)
        super(Channel_attention, self).build(input_shape)

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(inputs)
        proj_key = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)

        outputs = tf.keras.backend.batch_dot(attention, proj_query)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.beta * outputs + inputs

        return outputs


class Position_attention(tf.keras.layers.Layer):

    def __init__(self, filters, units, **kwargs):
        self.units = units
        self.filters = filters
        super(Position_attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.query_conv = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=1)
        self.key_conv = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=1)
        self.value_conv = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=1)
        self.gamma = self.add_weight(name='gamma',
                                     shape=(self.units, ),
                                     initializer='zeros',
                                     trainable=True)
        super(Position_attention, self).build(input_shape)

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(self.query_conv(inputs))
        proj_query = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        proj_key = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                            input_shape[4]))(self.key_conv(inputs))
        energy = tf.keras.backend.batch_dot(proj_query, proj_key)
        attention = tf.keras.activations.softmax(energy)
        attention = tf.keras.backend.permute_dimensions(attention, (0, 2, 1))

        proj_value = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(self.value_conv(inputs))

        outputs = tf.keras.backend.batch_dot(proj_value, attention)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs


class Attention_Embedding(tf.keras.layers.Layer):

    def __init__(self, filters, units, activation=None, rate=None, **kwargs):
        self.units = units
        self.filters = filters
        self.activation = activation
        self.dropout = tf.keras.layers.Dropout(rate)
        super(Attention_Embedding, self).__init__(**kwargs)

    def build(self, input_shape):
        self.channel_conv = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1, 1, 4),
                                                   activation=self.activation)
        self.position_conv = tf.keras.layers.Conv3D(filters=self.filters, kernel_size=(1, 1, 4),
                                                    activation=self.activation)
        super(Attention_Embedding, self).build(input_shape)

    def call(self, inputs):
        channel_conv = self.channel_conv(Channel_attention(units=1)(inputs))
        channel_outputs = self.dropout(channel_conv)

        position_conv = self.position_conv(Position_attention(filters=self.filters, units=1)(inputs))
        position_outputs = self.dropout(position_conv)

        outputs = channel_outputs + position_outputs
        outputs = self.dropout(outputs)

        return outputs
