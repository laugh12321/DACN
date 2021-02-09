#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2月 03, 2021 

@file: attention.py
@desc: channel attention layer
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

    def __init__(self):
        super(Channel_attention, self).__init__()

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma',
                                     shape=1,
                                     initializer='zeros',
                                     trainable=True)

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
        outputs = tf.keras.layers.multiply([self.gamma, outputs])
        outputs = tf.keras.layers.add([outputs, inputs])

        return outputs
