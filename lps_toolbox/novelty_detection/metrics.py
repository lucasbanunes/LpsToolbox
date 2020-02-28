"""Metrics to deal with the novelty_analysis package
Author: Lucas Barra de Aguiar Nunes"""
import numpy as np
from sklearn.metrics import recall_score

def sp_index_multiclass(y_true, y_pred, labels = None):
    n_classes = len(np.unique(y_true))
    exponent = 1/n_classes
    recall = recall_score(y_true, y_pred, labels = labels, average = None)
    sp = np.sqrt((np.sum(recall)/n_classes)*(np.power(np.prod(recall), exponent)))
    return sp

def sp_index_binary(y_true, y_pred, correct_cls, labels = None):
    y_true = np.where(y_true == correct_cls, 1, 0)
    y_pred = np.where(y_pred == correct_cls, 1, 0)
    n_classes = len(np.unique(y_true))
    exponent = 1/n_classes
    recall = recall_score(y_true, y_pred, labels = labels, average = None)
    sp = np.sqrt((np.sum(recall)/n_classes)*(np.power(np.prod(recall), exponent)))
    return sp
