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


class Channel_attention(tf.keras.layers.Layer):
    """
    3D implementation of Channel attention:
    Fu, Jun, et al. "Dual attention network for scene segmentation."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(self, **kwargs):
        super(Channel_attention, self).__init__(**kwargs)

    def call(self, inputs):
        in_shp = inputs.get_shape().as_list()

        # first Batch
        channel_C1 = tf.keras.layers.Reshape((in_shp[1] * in_shp[2] * in_shp[3], in_shp[4]))(inputs)

        # Second Batch
        channel_C2 = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 2, 1)))(channel_C1)
        channel_C2 = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([channel_C1, channel_C2])
        channel_C2 = tf.keras.activations.softmax(channel_C2)

        channel_C3 = tf.keras.layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([channel_C2, channel_C1])
        channel_C3 = tf.keras.layers.Reshape((in_shp[1], in_shp[2], in_shp[3], in_shp[4]))(channel_C3)

        outputs = tf.keras.layers.multiply([channel_C3, inputs])

        return outputs


class Position_attention(tf.keras.layers.Layer):
    """
    3D implementation of Position attention:
    Fu, Jun, et al. "Dual attention network for scene segmentation."
    Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, postion_attention_input):
        # Getting the Shape of the inputs
        in_shp = postion_attention_input.get_shape().as_list()

        C1 = tf.keras.layers.Conv3D(filters=int(in_shp[4]), kernel_size=1, strides=(1, 1, 1))(postion_attention_input)
        C1_shp = C1.get_shape().as_list()

        # first Batch
        F1_HWDxC = tf.reshape(C1, [-1, C1_shp[1] * C1_shp[2] * C1_shp[3], C1_shp[4]])

        # Second Batch
        F2_CxHWD = tf.transpose(F1_HWDxC, perm=[0, 2, 1])
        F2_CxHWD = tf.matmul(F1_HWDxC, F2_CxHWD)
        F2_CxHWD = tf.keras.activations.softmax(F2_CxHWD)

        # third Batch
        C2 = tf.keras.layers.Conv3D(filters=in_shp[4], kernel_size=1, strides=(1, 1, 1))(postion_attention_input)
        F3_HWDxC = tf.reshape(C2, [-1, in_shp[1] * in_shp[2] * in_shp[3], in_shp[4]])
        F3xF2 = tf.matmul(F2_CxHWD, F3_HWDxC)
        F3 = tf.reshape(F3xF2, [-1, in_shp[1], in_shp[2], in_shp[3], in_shp[4]])

        postion_attention_output = tf.keras.layers.Multiply()([postion_attention_input, F3])
        postion_attention_output = tf.keras.layers.Conv3D(filters=in_shp[4], kernel_size=1, strides=(1, 1, 1))(
            postion_attention_output)

        return postion_attention_output
