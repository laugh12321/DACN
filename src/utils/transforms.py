#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Jan 29, 2021

@file: transforms.py
@desc: Module containing all the transformations that can be done on a datasets.
@author: laugh12321
@contact: laugh12321@vip.qq.com
"""
import abc
import numpy as np
from typing import List, Dict

from src.model import enums
from src.model.models import rnn_supervised, pixel_based_cnn, \
    pixel_based_dcae, pixel_based_fnnc, \
    pixel_based_dacn, pixel_based_dacae


class BaseTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Each subclass should implement this method.
        :param args: Arbitrary list of arguments.
        :param kwargs: Arbitrary dictionary of arguments.
        """
        pass


class SpectralTransform(BaseTransform):
    def __init__(self, **kwargs):
        """
        Initializer of the spectral transformation.
        """
        super().__init__()

    def __call__(self, samples: np.ndarray,
                 labels: np.ndarray) -> List[np.ndarray]:
        """
        Transform 1D samples along the spectral axis.
        Only the spectral features are present for each sample in the datasets.

        :param samples: Input samples that will undergo transformation.
        :param labels: Class value for each samples.
        :return: List containing the transformed samples and the class labels.
        """
        return [np.expand_dims(samples.astype(np.float32), -1), labels]


class MinMaxNormalize(BaseTransform):
    def __init__(self, min_: float, max_: float):
        """
        Normalize each sample.

        :param min_: Minimum value of features.
        :param max_: Maximum value of features.
        """
        super().__init__()
        self.min_ = min_
        self.max_ = max_

    def __call__(self, samples: np.ndarray, labels: np.ndarray) -> List[
        np.ndarray]:
        """"
        Perform min-max normalization on the passed samples.

        :param samples: Input samples that will undergo normalization.
        :param labels: Class values for each sample.
        :return: List containing the normalized samples and the class labels.
        """
        return [(samples - self.min_) / (self.max_ - self.min_), labels]


def apply_transformations(data: Dict,
                          transformations: List[BaseTransform]) -> Dict:
    """
    Apply each transformation from provided list

    :param data: Dictionary with 'data' and 'labels' keys holding np.ndarrays
    :param transformations: List of transformations
    :return: Transformed data, in the same format as input
    """
    for transformation in transformations:
        data[enums.Dataset.DATA], data[enums.Dataset.LABELS] = transformation(
            data[enums.Dataset.DATA], data[enums.Dataset.LABELS])
    return data


class ExtractCentralPixelSpectrumTransform(BaseTransform):
    def __init__(self):
        """
        Extract central pixel from each spatial sample.
        :param neighborhood_size: The spatial size of the patch.
        """
        super().__init__()

    def __call__(self, samples: np.ndarray,
                 labels: np.ndarray) -> List[np.ndarray]:
        """"
        Transform the labels for unsupervised unmixing problem.
        The label is the central pixel of each sample patch.
        :param samples: Input samples.
        :param labels: Central pixel of each sample.
        :return: List containing the input samples
            and its targets as central pixel.
        """
        labels = np.squeeze(samples)
        return [samples, labels]


UNMIXING_TRANSFORMS = {
    rnn_supervised.__name__: [SpectralTransform],

    pixel_based_cnn.__name__: [SpectralTransform],

    pixel_based_dcae.__name__:
        [ExtractCentralPixelSpectrumTransform,
         SpectralTransform],

    pixel_based_fnnc.__name__: [SpectralTransform],
    
    pixel_based_dacn.__name__: [SpectralTransform],

    pixel_based_dacae.__name__:
        [ExtractCentralPixelSpectrumTransform,
         SpectralTransform]
}