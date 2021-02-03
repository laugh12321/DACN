#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2月 03, 2021 

@file: attention.py
@desc: 
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import tensorflow as tf


class Position_attention(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, Channel_attention_input):
        in_shp = Channel_attention_input.get_shape().as_list()

        # --first-Batch
        channel_C1 = tf.reshape(Channel_attention_input, [-1, in_shp[1] * in_shp[2] * in_shp[3], in_shp[4]])

        # --Second-Batch
        channel_C2 = tf.transpose(channel_C1, perm=[0, 2, 1])
        channel_C2 = tf.matmul(channel_C1, channel_C2)
        channel_C2 = tf.keras.activations.softmax(channel_C2)

        channel_C3 = tf.matmul(channel_C2, channel_C1)
        channel_C3 = tf.reshape(channel_C3, [-1, in_shp[1], in_shp[2], in_shp[3], in_shp[4]])

        Channel_attention_output = tf.keras.layers.Multiply()([channel_C3, Channel_attention_input])
        Channel_attention_output = tf.keras.layers.Conv3D(filters=in_shp[4], kernel_size=1, strides=(1, 1, 1))(
            Channel_attention_output)

        return Channel_attention_output
