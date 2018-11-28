"""
Metrics to assess performance on classification and novelty detection tasks

Authors: Pedro Henrique Braga Lisboa <pedro.lisboa@lps.ufrj.br>
"""

from __future__ import division

import sklearn
import numpy as np

from keras import backend as K
from keras.utils import to_categorical, get_custom_objects

def sp_index(y_true, y_pred):
    """Sum-Product Index Score

    Parameters
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix / tensorflow sparse tensor
        Ground truth (correct) labels.
    y_pred : 1d array-like, or label indicator array / sparse matrix / tensorflow tensor
        Predicted labels, as returned by a classifier.

    Returns
    -------
    sp : float
    """

    y_true_type = _check_type(y_true)
    y_pred_type = _check_type(y_pred)

    if y_true_type != y_pred_type:
        raise TypeError
    elif y_true_type == 'numpy':
        if y_pred.ndim == 1:
            y_pred = to_categorical(y_pred, np.unique(y_true).shape[0])
        if y_true.ndim == 1:  # turn 1d array to sparse matrix
            y_true = to_categorical(y_true, np.unique(y_true).shape[0])

        sp = _NpMetrics.sp_index(y_true, y_pred)
    elif y_true_type == 'tensorflow':
        sp = _TfMetrics.sp_index(y_true, y_pred)
    else:
        raise TypeError

    return sp


def recall_score(y_true, y_pred, average=None, clf_idx=None):
    """Sum-Product Index Score

        :param y_true : 1d array-like, or label indicator array / sparse matrix / tensorflow sparse tensor
                        Ground truth (correct) labels.
        :param y_pred : 1d array-like, or label indicator array / sparse matrix / tensorflow tensor
                        Predicted labels, as returned by a classifier.
        :param average:

        :returns recall : float
    """
    y_true_type = _check_type(y_true)
    y_pred_type = _check_type(y_pred)

    if y_true_type != y_pred_type:
        raise TypeError
    elif y_true_type == 'numpy':
        if y_pred.ndim == 1:
            y_pred = to_categorical(y_pred, np.unique(y_true).shape[0])
        if y_true.ndim == 1:  # turn 1d array to sparse matrix
            y_true = to_categorical(y_true, np.unique(y_true).shape[0])

        recall = _NpMetrics.recall_score(y_true, y_pred)
    elif y_true_type == 'tensorflow':
        recall = _TfMetrics.recall_score(y_true, y_pred, clf_idx)
    else:
        raise TypeError

    return recall


def precision_score(y_true, y_pred):
    raise NotImplementedError


def f1_score(y_true, y_pred):
    raise NotImplementedError


def _check_type(array):
    if isinstance(array, np.ndarray):
        type_obj = 'numpy'
    elif K.is_tensor(array):
        type_obj = 'tensorflow'
    else:
        type_obj = 'unknown'
    return type_obj


class _TfMetrics:
    def __init__(self):
        pass

    @staticmethod
    def sp_index(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        num_classes = K.int_shape(y_pred)[1]  # y_true returns (None, None) for int_shape
                                              # y_pred returns (None, num_classes)

        true_positives = K.sum(K.cast(y_true * K.one_hot(K.argmax(y_pred, axis=1), num_classes), dtype='float32'),
                               axis=0)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
        recall = true_positives / (possible_positives + K.epsilon())
        sp = K.sqrt(K.mean(recall) * K.prod(K.pow(recall, 1 / num_classes)))

        return sp

    @staticmethod
    def recall_score(y_true, y_pred, cls_idx):
        true_positives = K.sum(K.cast(y_true * K.one_hot(K.argmax(y_pred, axis=1), num_classes), dtype='float32'),
                               axis=0)
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)
        recall = true_positives / (possible_positives + K.epsilon())
        return recall[cls_idx]

    @staticmethod
    def accuracy_score(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        num_classes = K.int_shape(y_pred)[1]  # y_true returns (None, None) for int_shape
        # y_pred returns (None, num_classes)

        true_positives = K.sum(K.cast(y_true * K.one_hot(K.argmax(y_pred, axis=1), num_classes), dtype='float32'))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())

        return K.mean(recall)


class _NpMetrics:
    def __init__(self):
        pass

    @staticmethod
    def np_accuracy_score(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        num_classes = y_true.shape[1]
        recall = recall_score(y_true, y_pred, None)
        return np.sum(recall) / num_classes

    @staticmethod
    def sp_index(y_true, y_pred):
        """

        :param y_true:
        :param y_pred:
        :return:
        """
        num_classes = y_true.shape[1]
        recall = recall_score(y_true, y_pred, None)
        sp = np.sqrt(np.sum(recall) / num_classes *
                     np.power(np.prod(recall), 1.0 / float(num_classes)))
        return sp

    @staticmethod
    def recall_score(y_true, y_pred, average=None):
        recall = sklearn.metrics.recall_score(y_true, y_pred, average=average)
        return recall

get_custom_objects().update({"sp_index": sp_index})
