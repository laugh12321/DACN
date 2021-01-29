#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: performance_metrics.py
@desc: All metrics that are calculated on the model's output.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import numpy as np
import tensorflow as tf
from typing import Dict, List

from src.model.models import unmixing_rnn_supervised, \
    unmixing_pixel_based_dcae, unmixing_cube_based_dcae, \
    unmixing_cube_based_cnn, unmixing_pixel_based_cnn


def convert_to_tensor(metric_function):
    def wrapper(y_true: np.ndarray, y_pred: np.ndarray):
        if not isinstance(y_true, tf.Tensor):
            y_true = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
        if not isinstance(y_pred, tf.Tensor):
            y_pred = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
        with tf.Session() as session:
            return metric_function(y_true=y_true,
                                   y_pred=y_pred).eval(session=session)

    return wrapper


def spectral_information_divergence_loss(y_true: tf.Tensor,
                                         y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the spectral information divergence loss,
    which is based on the divergence in information theory.

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional
    autoencoders in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations and
    Remote Sensing 13 (2020): 567-576.

    :param y_true: Labels as two dimensional abundances or original
        input array of shape: [n_samples, n_classes], [n_samples, n_bands].
    :param y_pred: Predicted abundances or reconstructed input array of shape:
    [n_samples, n_classes], [n_samples, n_bands].
    :return: The spectral information divergence loss.
    """
    y_true_row_sum = tf.reduce_sum(y_true, 1)
    y_pred_row_sum = tf.reduce_sum(y_pred, 1)
    y_true = y_true / tf.reshape(y_true_row_sum, (-1, 1))
    y_pred = y_pred / tf.reshape(y_pred_row_sum, (-1, 1))
    y_true, y_pred = tf.keras.backend.clip(y_true,
                                           tf.keras.backend.epsilon(), 1), \
                     tf.keras.backend.clip(y_pred,
                                           tf.keras.backend.epsilon(), 1)
    loss = tf.reduce_sum(y_true * tf.log(y_true / y_pred)) + \
           tf.reduce_sum(y_pred * tf.log(y_pred / y_true))
    return loss


def average_angle_spectral_mapper(y_true: tf.Tensor,
                                  y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the dcae average angle spectral mapper value.

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes] or original input pixel
        and its reconstruction of shape: [n_samples, n_bands].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes]
        or original input pixel and
        its reconstruction of shape: [n_samples, n_bands].
    :return: The root-mean square abundance angle distance error.
    """
    numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    y_true_len = tf.sqrt(tf.reduce_sum(tf.square(y_true), 1))
    y_pred_len = tf.sqrt(tf.reduce_sum(tf.square(y_pred), 1))
    denominator = tf.multiply(y_true_len, y_pred_len)
    loss = tf.reduce_mean(tf.acos(
        tf.clip_by_value(numerator / denominator, -1, 1)))
    return loss


def dcae_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the custom dcae root-mean square error,
    which measures the similarity between the original abundance
    fractions and the predicted ones.

    Khajehrayeni, Farshid, and Hassan Ghassemian.
    "Hyperspectral unmixing using deep convolutional autoencoders
    in a supervised scenario."
    IEEE Journal of Selected Topics in Applied Earth Observations
    and Remote Sensing 13 (2020): 567-576.

    :param y_true: Labels as two dimensional abundances
    array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error.
    """
    return tf.reduce_mean(tf.sqrt(tf.reduce_mean(
        tf.square(y_pred - y_true), axis=1)))


def overall_rms_abundance_angle_distance(y_true: tf.Tensor,
                                         y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the cnn root-mean square abundance angle distance,
    which measures the similarity between the original abundance fractions
    and the predicted ones. Taken from cnn paper.
    It utilizes the inverse of cosine function at the range [0, pi],
    which means that the domain of arccos is in the range [-1; 1],
    that is why the "tf.clip_by_value" method is used.
    For the identical abundances the numerator / denominator is 1 and
    arccos(1) is 0, which resembles the perfect score.

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square abundance angle distance error.
    """
    numerator = tf.reduce_sum(tf.multiply(y_true, y_pred), 1)
    y_true_len = tf.sqrt(tf.reduce_sum(tf.square(y_true), 1))
    y_pred_len = tf.sqrt(tf.reduce_sum(tf.square(y_pred), 1))
    denominator = tf.multiply(y_true_len, y_pred_len)
    loss = tf.sqrt(tf.reduce_mean(tf.square(tf.acos(
        tf.clip_by_value(numerator / denominator, -1, 1)))))
    return loss


def sum_per_class_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the sum of per class root-mean square error.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The sum of per class root-mean square error.
    """
    return tf.reduce_sum(per_class_rmse(y_true=y_true, y_pred=y_pred))


def per_class_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the per class root-mean square error vector.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error vector.
    """
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2, 0))


def cnn_rmse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the custom cnn root-mean square error, which measures the
    similarity between the original abundance fractions and the predicted ones.

    Zhang, Xiangrong, Yujia Sun, Jingyan Zhang, Peng Wu, and Licheng Jiao.
    "Hyperspectral unmixing via deep convolutional neural networks."
    IEEE Geoscience and Remote Sensing Letters 15, no. 11 (2018): 1755-1759.

    :param y_true: Labels as two dimensional abundances
        array of shape: [n_samples, n_classes].
    :param y_pred: Predicted abundances of shape: [n_samples, n_classes].
    :return: The root-mean square error.
    """
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))


UNMIXING_TRAIN_METRICS = {
    unmixing_pixel_based_dcae.__name__: [spectral_information_divergence_loss],
    unmixing_cube_based_dcae.__name__: [spectral_information_divergence_loss],

    unmixing_pixel_based_cnn.__name__: [cnn_rmse,
                                        overall_rms_abundance_angle_distance,
                                        sum_per_class_rmse],
    unmixing_cube_based_cnn.__name__: [cnn_rmse,
                                       overall_rms_abundance_angle_distance,
                                       sum_per_class_rmse],

    unmixing_rnn_supervised.__name__: [cnn_rmse,
                                       overall_rms_abundance_angle_distance,
                                       sum_per_class_rmse]
}

UNMIXING_TEST_METRICS = {
    'aRMSE': dcae_rmse,
    'aSAM': average_angle_spectral_mapper,
    'overallRMSE': cnn_rmse,
    'rmsAAD': overall_rms_abundance_angle_distance,
    'perClassSumRMSE': sum_per_class_rmse
}

UNMIXING_LOSSES = {
    unmixing_pixel_based_dcae.__name__: spectral_information_divergence_loss,
    unmixing_cube_based_dcae.__name__: spectral_information_divergence_loss,

    unmixing_pixel_based_cnn.__name__: 'mse',
    unmixing_cube_based_cnn.__name__: 'mse',

    unmixing_rnn_supervised.__name__: 'mse'
}


def calculate_unmixing_metrics(**kwargs) -> Dict[str, List[float]]:
    """
    Calculate the metrics for unmixing problem.

    :param kwargs: Additional keyword arguments.
    """
    model_metrics = {}
    print(kwargs['y_pred'].shape)
    for f_name, f_metric in UNMIXING_TEST_METRICS.items():
        model_metrics[f_name] = [float(convert_to_tensor(f_metric)
                                       (y_true=kwargs['y_true'],
                                        y_pred=kwargs['y_pred']))]

    for class_idx, class_rmse in enumerate(convert_to_tensor(per_class_rmse)(
            y_true=kwargs['y_true'], y_pred=kwargs['y_pred'])):
        model_metrics[f'class{class_idx}RMSE'] = [float(class_rmse)]
    if kwargs['endmembers'] is not None:
        # Calculate the reconstruction RMSE and SID losses:
        x_pred = np.matmul(kwargs['y_pred'], kwargs['endmembers'].T)
        model_metrics['rRMSE'] = [float(convert_to_tensor(dcae_rmse)
                                        (y_true=kwargs['x_true'],
                                         y_pred=x_pred))]
        model_metrics['rSID'] = [float(convert_to_tensor(
            spectral_information_divergence_loss)
                                       (y_true=kwargs['x_true'],
                                        y_pred=x_pred))]
    return model_metrics