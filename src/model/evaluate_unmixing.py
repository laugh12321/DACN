#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: evaluate_unmixing.py
@desc: Perform the inference of the unmixing models on the test datasets.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import os
import numpy as np
import tensorflow as tf

import src.model.enums as enums
from src.utils import io, transforms
from src.evaluation.time_metrics import timeit
from src.utils.transforms import UNMIXING_TRANSFORMS
from src.utils.utils import get_central_pixel_spectrum
from src.evaluation.performance_metrics import UNMIXING_EVAL_METRICS, \
    calculate_unmixing_metrics



def evaluate(data,
             model_name: str,
             dest_path: str,
             batch_size: int):
    """
    Function for evaluating the trained model for the unmixing problem.

    :param model_name: model name.
    :param data: The data dictionary containing the subset for testing.
    :param dest_path: Directory in which to store the calculated metrics,
            and Path to the model.
    :param batch_size: Size of the batch for inference.
    """
    model = tf.keras.models.load_model(
        os.path.join(dest_path, 'model.h5'), compile=True,
        custom_objects={metric.__name__: metric for metric in
                        UNMIXING_EVAL_METRICS[model_name]})

    test_dict = data[enums.Dataset.TEST]

    min_, max_ = io.read_min_max(os.path.join(dest_path, 'min-max.csv'))

    transformations = [transforms.MinMaxNormalize(min_=min_, max_=max_)]
    transformations += [t() for t in UNMIXING_TRANSFORMS[model_name]]
    test_dict_transformed = transforms.apply_transformations(test_dict.copy(),
                                                             transformations)

    predict = timeit(model.predict)
    y_pred, inference_time = predict(
        test_dict_transformed[enums.Dataset.DATA],
        batch_size=batch_size)

    model_metrics = calculate_unmixing_metrics(**{
        'y_pred': y_pred,
        'y_true': test_dict[enums.Dataset.LABELS],
        'x_true': get_central_pixel_spectrum(
            test_dict_transformed[enums.Dataset.DATA])
    })

    model_metrics['inference_time'] = [inference_time]
    io.save_metrics(dest_path=dest_path,
                    file_name=enums.Experiment.INFERENCE_METRICS,
                    metrics=model_metrics)
