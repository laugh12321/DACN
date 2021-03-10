#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2月 03, 2021 

@file: attention.py
@desc: Dual attention module
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import tensorflow as tf


class Channel_attention(tf.keras.layers.Layer):
    """ Channel attention module """
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(Channel_attention, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        super(Channel_attention, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

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
        outputs = self.gamma * outputs + inputs

        return outputs


class Position_attention(tf.keras.layers.Layer):
    """ Position attention module """
    def __init__(self,
                 ratio = 8,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(Position_attention, self).__init__(**kwargs)
        self.ratio = ratio
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        super(Position_attention, self).build(input_shape)
        self.query_conv = tf.keras.layers.Conv3D(filters=input_shape[-1] // self.ratio, 
                                                 kernel_size=(1, 1, 1), use_bias=False, 
                                                 kernel_initializer='he_normal')
        self.key_conv = tf.keras.layers.Conv3D(filters=input_shape[-1] // self.ratio, 
                                               kernel_size=(1, 1, 1), use_bias=False, 
                                               kernel_initializer='he_normal')
        self.value_conv = tf.keras.layers.Conv3D(filters=input_shape[-1], kernel_size=(1, 1, 1),
                                                 use_bias=False, kernel_initializer='he_normal')
        self.gamma = self.add_weight(shape=(1,),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        input_shape = inputs.get_shape().as_list()

        proj_query = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4] // self.ratio))(self.query_conv(inputs))
        proj_query = tf.keras.backend.permute_dimensions(proj_query, (0, 2, 1))
        proj_key = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                            input_shape[4] // self.ratio))(self.key_conv(inputs))
        energy = tf.keras.backend.batch_dot(proj_key, proj_query)
        attention = tf.keras.activations.softmax(energy)

        proj_value = tf.keras.layers.Reshape((input_shape[1] * input_shape[2] * input_shape[3],
                                              input_shape[4]))(self.value_conv(inputs))

        outputs = tf.keras.backend.batch_dot(attention, proj_value)
        outputs = tf.keras.layers.Reshape((input_shape[1], input_shape[2], input_shape[3],
                                           input_shape[4]))(outputs)
        outputs = self.gamma * outputs + inputs

        return outputs


def cbam_block(cbam_feature, ratio=8):
    """
    Contains the implementation of Convolutional Block Attention Module(CBAM) block.
    As described in https://arxiv.org/abs/1807.06521.
    """

    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)

    return cbam_feature


def channel_attention(input_feature, ratio=8):

    channel = input_feature.get_shape().as_list()[-1]

    shared_layer_one = tf.keras.layers.Dense(channel//ratio,
                                             activation='relu',
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')
    shared_layer_two = tf.keras.layers.Dense(channel,
                                             kernel_initializer='he_normal',
                                             use_bias=True,
                                             bias_initializer='zeros')

    avg_pool = tf.keras.layers.GlobalAveragePooling3D()(input_feature)    
    avg_pool = tf.keras.layers.Reshape((1,1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = tf.keras.layers.GlobalMaxPooling3D()(input_feature)
    max_pool = tf.keras.layers.Reshape((1,1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool,max_pool])
    cbam_feature = tf.keras.layers.Activation('sigmoid')(cbam_feature)

    return tf.keras.layers.multiply([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    channel = input_feature.get_shape().as_list()[-1]
    cbam_feature = input_feature

    avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(cbam_feature)
    max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(cbam_feature)
    concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])
    cbam_feature = tf.keras.layers.Conv3D(filters = 1, kernel_size=kernel_size,
                                          strides=1, padding='same', activation='sigmoid',
                                          kernel_initializer='he_normal', use_bias=False)(concat)	
        
    return tf.keras.layers.multiply([input_feature, cbam_feature])
    