"""This module contains base functions for dealing with the analysis of novelty detection with Neural Networks as classifiers
Author: Lucas Barra de Aguiar Nunes"""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, fbeta_score, accuracy_score, precision_score, confusion_matrix
from lps_toolbox.metrics.classification import sp_index
from lps_toolbox.novelty_detection.metrics import sp_index_multiclass, sp_index_binary


def create_threshold(quantity, 
                     interval):
        """
        creates a threshold array for the model using np.linspace
        
        Parameters

        quantity: iterable
            each position has the number of values for each interval in the same position
        interval: iterable
            each position has a len 2 iterable with the interval where to extract the corresponding quantity

        Return

            threshold: numpy.ndarray
        """
        threshold = np.array(list())
        for q, i in zip(quantity, interval):
            threshold = np.concatenate((threshold, np.linspace(i[0], i[1], q)))
        return threshold

def get_results(predictions,
                target,
                real_cls_values, 
                novelty_class,
                threshold,
                save_csv = False, 
                filepath = './',
                filename = None):
        """
        creates a data frame with the events as rows and the output of the neurons in the output layer, 
        the classification for each threshold and the target as columns

        Parameters

        predictions: numpy array
            prediction set with events on lines ans neurons on columns

        target: numpy array
            correct target of the prediction set (can't be one hot encoded)
            Example: target = [1,0,0,2,3, ..., 3]

        real_cls_values: 1-D array like
            list which the position i has the correct number of the class i in the network

        novelty_class: int
            class that will be treated as novelty
            
        threshold: numpy array
            array with one or multiples values of threshold to be tested    

        save _csv: boolean
            if true saves the data frame as a .csv file

        filepath: string
            where to save the .csv file

        Return

            novelty_frame: pandas.DataFrame
        """

        classf = _get_classf_matrix(predictions, real_cls_values, novelty_class, threshold)
        target = target.reshape(-1,1)
        data = np.concatenate((predictions, classf, target), axis = 1)
        outer_level = ['Neurons', 'Neurons', 'Neurons', 'Target']
        inner_level = ['out0', 'out1', 'out2', 'T']
        for t in threshold:
            outer_level.insert(-1, 'Classification')
            inner_level.insert(-1, t)
        novelty_frame = pd.DataFrame(data, columns = pd.MultiIndex.from_arrays([outer_level, inner_level]))
        if save_csv:
            filepath = filepath + '/' +filename
            novelty_frame.to_csv(filepath, index = False)
        return novelty_frame

def read_results_csv(filepath):
    """
    reads the csv file generated from get_results() and returns the DataFrame

    Parameters

    filepath: string
        path to the file location
    
    Return
        novelty_frame: pandas.DataFrame
    """
    novelty_frame = pd.read_csv(filepath, header = 1)
    inner_level = list(novelty_frame.columns)
    outer_level = ['Neurons', 'Neurons', 'Neurons', 'Target']
    for i in range (len(inner_level)-4):
        outer_level.insert(-1, 'Classification')
    novelty_frame.columns = pd.MultiIndex.from_arrays([outer_level, inner_level])
    return novelty_frame

def get_novelty_eff(results_frame,
                    novelty_class,
                    average = None, 
                    beta = 1, 
                    normalize = True, 
                    save_csv = False,
                    filepath = './',
                    filename = None,
                    verbose = False):
    """
    returns a data frame with several parameters of efficiency for novelty detection

    Parameters

    results_frame: pandas.DataFrame
        frame obtained from get_results function

    novelty_class
        class treated as novelty
    
    average: None or string
        parameter defined for sklearn.metrics.precision_recall_fscore_support
    
    beta: float or int
        beta value to be used in the evaluation of f-beta
    
    normalize: boolean
        defines normalization of the precision metric
    
    verbose: boolean
        if true gives output of the function's activity
    
    Return
        eff_frame: pandas.DataFrame
    """
    y_true = np.where(results_frame.loc[:, 'Target'].values == novelty_class, 1, 0)
    novelty_matrix = np.array(get_novelty_from_frame(results_frame, novelty_class), dtype = float)
    thresh = results_frame.loc[:, 'Classification'].columns.values
    precision = _get_precision(y_true, novelty_matrix, average=average)
    recall = _get_recall(y_true, novelty_matrix, average=average)
    fbeta = _get_fbeta(y_true, novelty_matrix, beta, average=average)
    if average:
        sp = _get_sp_index(y_true, novelty_matrix)
        acc = _get_accuracy(y_true, novelty_matrix, normalize)
        novelty_rate = class_detection_rate(y_true, novelty_matrix)
        if verbose:
            print(f'Precision array shape is {precision.shape} and it is:\n{precision}')
            print(f'Recall array shape is {recall.shape} and it is:\n{recall}')
            print(f'Fbeta array shape is {fbeta.shape} and it is:\n{fbeta}')
            print(f'Accuracy array shape is {acc.shape} and it is:\n{acc}')
            print(f'SP array shape is {sp.shape} and it is:\n{sp}')
        data = np.concatenate((acc.reshape(-1,1), precision.reshape(-1,1), recall.reshape(-1,1), fbeta.reshape(-1,1), sp.reshape(-1,1), novelty_rate.reshape(-1,1)), axis=1)
        nov_eff_frame = pd.DataFrame(data, index = thresh, columns = ['Accuracy', 'Precision', 'Recall', 'Fbeta', 'SP', 'Novelty Rate'])
        nov_eff_frame.index.name = 'Threshold'
    else:
        if verbose:
            print(f'y_true array shape is {y_true.shape} and it is:\n{y_true}')
            print(f'novelty_matrix.shape is {novelty_matrix.shape} and it is:\n{novelty_matrix}')
            print(f'Precision array shape is {precision.shape} and it is:\n{precision}')
            print(f'Recall array shape is {recall.shape} and it is:\n{recall}')
            print(f'Fbeta array shape is {fbeta.shape} and it is:\n{fbeta}')
        data = np.concatenate((precision[0].reshape(-1,1), recall[0].reshape(-1,1), fbeta[0].reshape(-1,1)), axis=1)
        data = np.concatenate((data, np.concatenate((precision[1].reshape(-1,1), recall[1].reshape(-1,1), fbeta[1].reshape(-1,1)), axis=1)), axis = 0)
        nov_eff_frame = pd.DataFrame(data, index = pd.MultiIndex.from_product([[0,1], thresh], names=('Novelty', 'Threshold')), columns = ['Precision', 'Recall', 'Fbeta'])
    if save_csv:
        filepath = filepath + '/' + filename
        nov_eff_frame.to_csv(filepath, index = True)
    return nov_eff_frame
    
def get_classf_eff(results_frame,
                   labels = None, 
                   average = None, 
                   beta = 1, 
                   normalize = True,
                   save_csv = False,
                   filepath = './',
                   filename = None,
                   verbose = False):
    """
    returns a data frame with several parameters of efficiency for classification and novelty detection

    Parameters

    results_frame: pandas.DataFrame
        frame obtained from get_results

    labels: 1-d array like
        contains the name of the classes or its numbers
    
    average: None or string
        parameter defined for sklearn.metrics.precision_recall_fscore_support
    
    beta: float or int
        beta value to be used in the evaluation of f-beta

    verbose: boolean
        if true gives outṕut of the function's activity

    Return
        eff_frame: pandas.DataFrame
    """
    y_true = results_frame.loc[:,'Target'].values
    thresh = results_frame.loc[:,'Classification'].columns.values
    classf_matrix = results_frame.loc[:,'Classification'].values
    classes_names = np.unique(y_true)
    precision = _get_precision(y_true, classf_matrix, average = average, labels = labels)
    recall = _get_recall(y_true, classf_matrix, average = average, labels = labels)
    fbeta = _get_fbeta(y_true, classf_matrix, beta = beta, average = average, labels = labels)
    sp_binary = _get_sp_binary(y_true, classf_matrix, classes = np.unique(y_true))
    data_creation = True
    if verbose:
        print(f'The precision array has shape {precision.shape} and it is:\n{precision}')
        print(f'The recall array has shape {recall.shape} and it is:\n{recall}')
        print(f'The precision array has shape {fbeta.shape} and it is:\n{fbeta}')
        print('Generating data frame')
    if average:
        sp = _get_sp_index(y_true, classf_matrix)
        acc = np.apply_along_axis(lambda x: accuracy_score(y_true, x, normalize), 0, classf_matrix)
        data = np.concatenate((acc.reshape(-1,1), precision.reshape(-1,1), recall.reshape(-1,1), fbeta.reshape(-1,1), sp.reshape(-1,1)), axis = 1)
        eff_frame = pd.DataFrame(data, index = thresh, columns = ['Accuracy', 'Precision', 'Recall', 'Fbeta', 'SP'])
        eff_frame.index.name = 'Threshold'
    else:
        for p, r, f, sp, c in zip(precision, recall, fbeta, sp_binary, classes_names):
            if data_creation:
                data = np.concatenate((p.reshape(-1,1), r.reshape(-1,1), f.reshape(-1,1), sp.reshape(-1,1)), axis = 1)
                data_creation = False
            else:
                d = np.concatenate((p.reshape(-1,1), r.reshape(-1,1), f.reshape(-1,1), sp.reshape(-1,1)), axis = 1)
                data = np.concatenate((data, d), axis = 0)
        eff_frame = pd.DataFrame(data, index = pd.MultiIndex.from_product([np.unique(y_true), thresh], names=('Class', 'Threshold')), columns = ['Precision', 'Recall', 'Fbeta', 'Binary SP Index'])
    if save_csv:
        filepath = filepath + '/' +filename
        eff_frame.to_csv(filepath, index = True)
    return eff_frame    

def get_novelty_from_frame(results_frame, novelty_class):
    """
    returns a novelty_matrix with True where novelty was detected and False where not

    Parameters

    results_frame: pandas.DataFrame
        frame obtained from get_results

    Return
        novelty_matrix: numpy.ndarray
    """
    return np.where(results_frame.loc[:, 'Classification'].values == novelty_class, True, False)

def _get_label_matrix(results_frame,
                      classes_names, 
                      novelty_index):
    """
    returns classification matrix and target array with the index of the classes instead of its names

    Parameters

    results_frame: pandas.DataFrame
        frame obtained from get_results
    classes_names: 1-D array_like
        1-D array_like with the name of the class in each of its index
    novelty_index: int
        index of the class that will be treated as an integer

    Return
        classf, target: numpy.ndarray
    """
    classes_names[novelty_index] = 'Nov'
    classf = results_frame['Classification'].values
    target = results_frame['Target'].values
    for i, c in enumerate(classes_names):
        classf = np.where(classf == c, i, classf)
        target = np.where(target == c, i, target)
    classf = np.array(classf, dtype = float)
    target = np.array(target, dtype = float)
    return classf, target
    
                    
def _get_novelty_matrix(predictions, threshold):
        """
        creates a boolean array with the value of novelty detection for the events as rows and for each value
        of the threshold as a column

        Parameters

        predictions: numpy array
            prediction set
        threshold: numpy array
            array with the threshold values
        
        Return
            novelty_matrix: numpy.ndarray, dtype = bool
        """

        novelty_matrix = np.empty(shape = (predictions.shape[0], threshold.shape[0]), dtype = bool)
        for i,t in enumerate(threshold):
            novelty_matrix[:, i] = (predictions < t).all(axis=1)
        
        return novelty_matrix
            
def _get_classf_matrix(predictions, 
                       real_cls_values,
                       novelty_class, 
                       threshold):
        """
        creates an array with the classfication for the events as rows and for each value of the threshold as a column

        Parameters

        predictions: numpy array
            prediction matrix
        real_cls_values: 1-D array like
            list which the position i has the correct number of the class i in the network
        novelty_class: int
            class that was treated as novelty
        threshold: numpy array
            array with the threshold values
        
        Return
            classf_matrix: numpy.ndarray, dtype=object
        """

        novelty_matrix = _get_novelty_matrix(predictions, threshold)
        stack = list()
        results = list()
        for i in np.argmax(predictions, axis = 1):
            results.append(real_cls_values[int(i)])
        for i in range(threshold.shape[0]):
            stack.append(results)
        classf_matrix = np.where(novelty_matrix, novelty_class, np.column_stack(tuple(stack)))
        return classf_matrix

        
    
def _get_target_matrix(target, 
                       classes_names, 
                       novelty_index):
        """
        creates an array with the target for the events as rows
        uses the classes names

        Parameters

        target: numpy array
            correct target of the prediction set
        classes_names: 1-D array like
            list which the position i has the correct number of the class i in the network
        novelty_index: int
            index of the class that will be treated as novelty

        Return
            target_matrix: numpy.ndarray, dtype=object
        """

        named_target = list()
        for t in target:
            named_target.append(classes_names[int(t)])
        target_matrix = np.array(named_target).reshape((-1, 1))
        return target_matrix
    
def _get_accuracy(y_true, pred_matrix, normalize=True, sample_weight=None):
    """
    implementation of sklearn.metrics.accuracy_score for the classification and novelty matrices
    the parameters work as the other function
    """
    return np.apply_along_axis(lambda x: accuracy_score(y_true, x, normalize, sample_weight), 0, pred_matrix)

def _get_sp_index(y_true, pred_matrix):
    """
    implementation of lps_toolbox.metrics.classification,sp_index for the novelty matrix
    the parameters work as the other function
    """
    y_true = y_true.flatten()
    if len(pred_matrix.shape) == 2:
        sp = np.array([])
        for prediction in pred_matrix.T:
            sp = np.append(sp, sp_index(y_true, prediction))
        
        return sp
    elif (len(pred_matrix.shape) == 1) or (pred_matrix.shape[1] == 1):
        return(y_true, pred_matrix.flatten())
    else:
        raise ValueError('The prediction matrix must be 2d or 1d')

def _get_recall(y_true, pred_matrix, labels=None, pos_label=1, average='binary', sample_weight=None):
    """
    implementation of sklearn.metrics.recall_score for the classification and novelty matrices
    the parameters work as the other function
    """
    recall = np.apply_along_axis(lambda x: recall_score(y_true, x, labels, pos_label, average, sample_weight), 0, pred_matrix)
    return recall

def _get_precision(y_true, pred_matrix, pos_label=1, average='binary', sample_weight=None, labels = None):
    """
    implementation of sklearn.metrics.precision_score for the classification and novelty matrices
    the parameters work as the other function
    """
    precision = np.apply_along_axis(lambda x: precision_score(y_true, x, labels = labels, average = average), 0, pred_matrix)
    return precision

def _get_fbeta(y_true, pred_matrix, beta, labels=None, pos_label=1, average='binary', sample_weight=None):
    """
    implementation of sklearn.metrics.fbeta_score for the classification and novelty matrices
    the parameters work as the other function
    """
    fbeta = np.apply_along_axis(lambda x: fbeta_score(y_true, x, beta, labels, pos_label, average, sample_weight), 0, pred_matrix)
    return fbeta

def _get_sp_binary(y_true, pred_matrix, classes):
    sp = list()
    for clas in classes:
        sp.append(np.apply_along_axis(lambda x: sp_index_binary(y_true, x, clas), 0, pred_matrix))
    return np.array(sp)

def class_detection_rate(y_true, pred_matrix, clas = None):
    """
    detection ratio of a determined class in comparison with the correct number of occurrences
    Parameters:

    y_true: numpy array
        Targeted classification if no "clas" parameter is provided the desired class to be calculated must be 1 and the others 0
        
    pred_matrix: numpy.array
        Classfication obtained from the model (events as rows) if no "clas" parameter is provided the desired class to be calculated must be 1 and the others 0
        

    clas: int
        Clas that will have its detection rate calculated
    
    Returns:
    detection_rate: int or 1d numpy array
    """
    if not (type(clas) == type(None)):
        y_true = np.where(y_true == clas, 1, 0)
        pred_matrix = np.where(pred_matrix == clas, 1, 0)

    detection_rate = np.array([])
    occurrences = np.sum(y_true == 1)
    for column in pred_matrix.T:
        pred_occurences = np.sum(column == 1)
        detection_rate = np.append(detection_rate, (pred_occurences/occurrences))
    return detection_rate