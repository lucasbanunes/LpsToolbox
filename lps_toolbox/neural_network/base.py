# coding=utf-8
"""Base classes and functions for building neural networks based estimators

Authors: Pedro Henrique Braga Lisboa <pedro.lisboa@lps.ufrj.br>
"""

from __future__ import print_function, division

import glob
import hashlib
import json
import os
import re
import warnings
from collections import OrderedDict
from itertools import cycle

import joblib
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn
from keras import Sequential

from sklearn.base import BaseEstimator, ClassifierMixin

from lps_toolbox.metrics.classification import sp_index, recall_score

MODEL_WEIGHTS_FILENAME = 'end_state'
MODEL_BEST_WEIGHTS_FILENAME = 'best_weights'
RECOVER_TRAINING_FILENAME = 'tmp_state'
MODEL_HISTORY_FILENAME = 'history.log'
CALLBACKS_INFO_FILENAME = 'checkpoint_policy.json'


def dump_checkpoint_config(trn_path, callbacks):
    callback_path = os.path.join(trn_path, CALLBACKS_INFO_FILENAME)
    if not os.path.exists(os.path.join(callback_path)):
        print(callbacks.to_json())
        fp = open(os.path.join(trn_path, CALLBACKS_INFO_FILENAME), 'wb')
        json.dump(callbacks.to_json(), fp)
        fp.close()
    else:
        raise NotImplementedError


class BaseNNClassifier(BaseEstimator, ClassifierMixin):
    """ dd
        Base class for building Neural Network based classifiers
        Works as an sklearn estimator
    """

    def __init__(self, input_shape=(None,), solver="adam", batch_size=32, epochs=200,
                 loss="categorical_crossentropy", metrics=None,
                 momentum=0.9, nesterov=True, decay=0.0,
                 beta_1=0.9, beta_2=0.999, epsilon=1e-08, learning_rate=0.001,
                 amsgrad=False, early_stopping=False, es_kwargs=None, model_checkpoint=True,
                 save_best=True, mc_kwargs=None, log_history=True, cachedir='./'):
        """

        :param input_shape:
        :param solver:
        :param batch_size:
        :param epochs:
        :param loss:
        :param metrics:
        :param momentum:
        :param nesterov:
        :param decay:
        :param beta_1:
        :param beta_2:
        :param epsilon:
        :param learning_rate:
        :param amsgrad:
        :param early_stopping:
        :param es_kwargs:
        :param model_checkpoint:
        :param save_best:
        :param mc_kwargs:
        :param log_history:
        :param cachedir:
        """
        if metrics is None:
            metrics = ["acc"]
        self.input_shape = input_shape
        self.solver = solver
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.decay = decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.amsgrad = amsgrad
        self.early_stopping = early_stopping
        self.es_kwargs = es_kwargs
        self.model_checkpoint = model_checkpoint
        self.save_best = save_best
        self.mc_kwargs = mc_kwargs
        self.log_history = log_history
        self.cachedir = cachedir
        self.history = None  # training history
        self.model = None  # model object after training

    def _build_checkpoints(self, filepath):
        if self.es_kwargs is None:
            es_kwargs = {"monitor": 'val_loss',
                         "min_delta": 0,
                         "patience": 10,
                         "verbose": 0,
                         "mode": 'auto',
                         "baseline": None,
                         "restore_best_weights": False}
        else:
            tmp_kwargs = {"monitor": 'val_loss',
                          "min_delta": 0,
                          "patience": 10,
                          "verbose": 0,
                          "mode": 'auto',
                          "baseline": None,
                          "restore_best_weights": False}

            for key in self.es_kwargs.keys():
                tmp_kwargs[key] = self.es_kwargs[key]
            es_kwargs = tmp_kwargs

        if self.mc_kwargs is None:
            mc_kwargs = {"monitor": 'val_loss',
                         "verbose": 0,
                         "save_weights_only": False,
                         "mode": 'auto',
                         "period": 1,
                         "save_best": self.save_best}
        else:
            tmp_kwargs = {"monitor": 'val_loss',
                          "verbose": 0,
                          "save_weights_only": False,
                          "mode": 'auto',
                          "period": 1,
                          "save_best": self.save_best}
            for key in self.mc_kwargs.keys():
                tmp_kwargs[key] = self.mc_kwargs[key]
            mc_kwargs = tmp_kwargs
        mc_kwargs["filepath"] = filepath

        callbacks = []
        if self.early_stopping:
            callbacks.append(build_early_stopping(**es_kwargs))
        if self.model_checkpoint:
            m_check, best_model = build_model_checkpoint(**mc_kwargs)
            callbacks.append(m_check)
            if best_model is not None:
                callbacks.append(best_model)
        if self.log_history:
            csvlog = {"type": "CSVLogger", "filename": os.path.join(filepath, MODEL_HISTORY_FILENAME)}
            callbacks.append(csvlog)

        callbacks_list = Callbacks()
        for callback in callbacks:
            callbacks_list.add(callback)
        self.callbacks = callbacks_list

        return callbacks_list

    def _build_topology(self):
        raise NotImplementedError

    def _build_model(self):
        optimizer = self.build_optimizer()

        layers = self._build_topology()
        model_params = NNParams(optimizer=optimizer,
                                layers=layers,
                                loss=self.loss,
                                metrics=self.metrics,
                                input_shape=self.input_shape)

        model = Sequential()
        for layer in layers:
            model.add(layer.to_keras_fn())
        model.compile(optimizer=optimizer.to_keras_fn(),
                      loss=self.loss,
                      metrics=self.metrics)
        return model, model_params

    def fit(self, X, y,
            n_inits=1,
            validation_split=0.0,
            validation_data=None,
            shuffle=True,
            verbose=0,
            class_weight=True,
            sample_weight=None,
            steps_per_epoch=None,
            validation_steps=None,
            cachedir='./'):
        """

        :param X:
        :param y:
        :param n_inits:
        :param validation_split:
        :param validation_data:
        :param shuffle:
        :param verbose:
        :param class_weight:
        :param sample_weight:
        :param steps_per_epoch:
        :param validation_steps:
        :param cachedir:
        :return:
        """
        if class_weight:
            class_weights = _get_gradient_weights(y)
        else:
            class_weights = None

        if n_inits < 1:
            warnings.warn("Number of initializations must be at least one."
                          "Falling back to one")
            n_inits = 1

        self.input_shape = X.shape[1:]
        model, model_p = self._build_model()
        trn_p = TrainParams(self.batch_size, self.epochs, verbose,
                            validation_split, validation_data, shuffle,
                            class_weight, sample_weight, steps_per_epoch, validation_steps)
        trn_path, last_init, trained = check_model_integrity(cachedir, model_p, trn_p)
        best_weights_path = os.path.join(trn_path, MODEL_BEST_WEIGHTS_FILENAME)

        if trained:
            print("Model trained, loading best weights")
            model.load_weights(best_weights_path)
        else:
            checkpoints = self._build_checkpoints(trn_path)
            dump_checkpoint_config(trn_path, checkpoints)
            if shuffle:
                X, y = sklearn.utils.shuffle(X, y)
            for init in range(last_init, n_inits):
                model.fit(x=X, y=y, batch_size=self.batch_size,
                          epochs=self.epochs, verbose=verbose,
                          callbacks=checkpoints.to_keras_fn(), validation_split=validation_split,
                          validation_data=validation_data, shuffle=True,
                          class_weight=class_weights, sample_weight=sample_weight,
                          initial_epoch=0, steps_per_epoch=steps_per_epoch,
                          validation_steps=validation_steps)

                optimizer = self.build_optimizer()  # Reset optimizer state
                model.compile(optimizer=optimizer.to_keras_fn(),  # Reset model state
                              loss=self.loss,
                              metrics=self.metrics)

                os.rename(os.path.join(trn_path, RECOVER_TRAINING_FILENAME),
                          os.path.join(trn_path, RECOVER_TRAINING_FILENAME + '_init_%i' % init))

            os.rename(os.path.join(trn_path, RECOVER_TRAINING_FILENAME + '_init_%i' % (n_inits - 1)),
                      os.path.join(trn_path, MODEL_WEIGHTS_FILENAME))
        model.load_weights(best_weights_path)
        self.history = pd.read_csv(os.path.join(trn_path, MODEL_HISTORY_FILENAME))
        self.model = model
        return self

    def predict(self, X):
        """

        :param X:
        :return:
        """
        return self.model.predict(X)

    def score(self, X, y, sample_weight=None, return_eff=True):
        """

        :param X:
        :param y:
        :param sample_weight:
        :return:
        """
        if y.ndim > 1:
            y = y.argmax(axis=1)

        out = self.predict(X)

        cat_out = out.argmax(axis=1)

        if return_eff:
            recall = recall_score(y, cat_out)
            scores = {'eff_%i' % cls_i:recall[cls_i] for cls_i in np.unique(y)}
            scores['sp'] = sp_index(y, cat_out)
            return scores

        return sp_index(y, cat_out)

    def plot_training(self, ax=None,
                      train_scores='all',
                      val_scores='all',
                      savepath=None):
        """

        :param ax:
        :param train_scores:
        :param val_scores:
        :param savepath:
        """
        if ax is None:
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(1, 1)
            loss_ax = ax
            score_ax = plt.twinx(ax)
        elif isinstance(ax, tuple):
            score_ax = ax[0]
            loss_ax = ax[1]
        else:
            score_ax, loss_ax = ax, ax

        history = self.history

        if val_scores is 'all':
            val_re = re.compile('val_')
            val_scores = map(lambda xs: xs.string,
                             filter(lambda xs: xs is not None,
                                    map(val_re.search, history.columns.values)))

        if train_scores is 'all':
            train_scores = history.columns.values[1:]  # Remove 'epoch' column
            train_scores = train_scores[~np.isin(train_scores, val_scores)]

        x = history['epoch']
        linestyles = ['-', '--', '-.', ':']
        ls_train = cycle(linestyles)
        ls_val = cycle(linestyles)

        loss_finder = re.compile('loss')
        for train_score, val_score in zip(train_scores, val_scores):
            if loss_finder.search(train_score) is None:
                score_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)
            else:
                loss_ax.plot(x, history[train_score], color="blue", linestyle=ls_train.next(), label=train_score)

            if loss_finder.search(val_score) is None:
                score_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)
            else:
                loss_ax.plot(x, history[val_score], color="red", linestyle=ls_val.next(), label=val_score)

        ax.legend()
        plt.show()

    def build_optimizer(self):
        """

        :return:
        """
        return _build_optimizer(self.solver, self.momentum, self.nesterov,
                                     self.decay, self.learning_rate, self.amsgrad,
                                     self.beta_1, self.beta_2, self.epsilon)


def build_early_stopping(monitor,
                         min_delta,
                         patience,
                         verbose,
                         mode,
                         baseline,
                         restore_best_weights):
    """

    :param monitor:
    :param min_delta:
    :param patience:
    :param verbose:
    :param mode:
    :param baseline:
    :param restore_best_weights:
    :return:
    """
    return {"type": "EarlyStopping",
            "monitor": monitor,
            "min_delta": min_delta,
            "patience": patience,
            "verbose": verbose,
            "mode": mode,
            "baseline": baseline,
            "restore_best_weights": restore_best_weights}


def _build_optimizer(solver,
                     momentum=0.9,
                     nesterov=True,
                     decay=0.0,
                     learning_rate=0.001,
                     amsgrad=False,
                     beta_1=0.9,
                     beta_2=0.999,
                     epsilon=1e-08):
    solver = solver.lower()

    optimizer = {}
    if solver not in ['sgd', 'adam']:
        raise NotImplementedError

    if solver == 'sgd':
        optimizer = {"type": "SGD",
                     "momentum": momentum,
                     "decay": decay,
                     "nesterov": nesterov}

    elif solver == 'adam':
        optimizer = {"type": "Adam",
                     "lr": learning_rate,
                     "beta_1": beta_1,
                     "beta_2": beta_2,
                     "epsilon": epsilon,
                     "decay": decay,
                     "amsgrad": amsgrad}

    return Optimizer(optimizer)


def build_model_checkpoint(filepath,
                           monitor,
                           verbose,
                           save_weights_only,
                           mode,
                           period,
                           save_best):
    """

    :param filepath:
    :param monitor:
    :param verbose:
    :param save_weights_only:
    :param mode:
    :param period:
    :param save_best:
    :return:
    """
    m_filepath = os.path.join(filepath, RECOVER_TRAINING_FILENAME)
    b_filepath = os.path.join(filepath, MODEL_BEST_WEIGHTS_FILENAME)

    m_check = {"type": "ModelCheckpoint",
               "filepath": m_filepath,
               "monitor": monitor,
               "verbose": verbose,
               "save_weights_only": save_weights_only,
               "mode": mode,
               "period": period}
    if save_best:
        best_check = {"type": "ModelCheckpoint",
                      "filepath": b_filepath,
                      "monitor": monitor,
                      "verbose": verbose,
                      "save_weights_only": save_weights_only,
                      "mode": mode,
                      "period": period,
                      "save_best_only": True}
    else:
        best_check = None

    return m_check, best_check


def _get_gradient_weights(y_train):
    if y_train.ndim > 1:
        y_train = y_train.argmax(axis=1)

    cls_indices, event_count = np.unique(np.array(y_train), return_counts=True)
    min_class = min(event_count)

    return {cls_index: float(min_class) / cls_count
            for cls_index, cls_count in zip(cls_indices, event_count)}


class TrainParams(object):
    """
        Params Storage and builder Obj for Neural Network Based Classifier training
    """
    def __init__(self, batch_size, epochs, verbose, validation_split, validation_data,
                 shuffle, class_weight, sample_weight, steps_per_epoch, validation_steps):
        self.verbose = verbose
        self.validation_steps = validation_steps
        self.steps_per_epoch = steps_per_epoch
        self.sample_weight = sample_weight
        self.class_weight = class_weight
        self.shuffle = shuffle
        self.validation_data = validation_data
        self.validation_split = validation_split
        self.epochs = epochs
        self.batch_size = batch_size

    def to_json(self):
        params = OrderedDict(batch_size=self.batch_size,
                             epochs=self.epochs,
                             verbose=self.verbose,
                             validation_split=self.validation_split,
                             validation_data=self.validation_data,
                             shuffle=self.shuffle,
                             class_weight=self.class_weight,
                             sample_weight=self.sample_weight,
                             steps_per_epoch=self.steps_per_epoch,
                             validation_steps=self.validation_steps)
        return params

    def get_param_hash(self):
        json_params = self.to_json()
        hash_params = hashlib.sha512(str(json_params)).hexdigest()
        return hash_params


def check_model_integrity(basefolder, model_p, trn_p):
    trn_hash = trn_p.get_param_hash()
    model_hash = model_p.get_param_hash()

    model_path = os.path.join(basefolder, model_hash)
    trn_path = os.path.join(model_path, trn_hash)
    best_weights_path = os.path.join(trn_path, MODEL_BEST_WEIGHTS_FILENAME)
    train_model_state = os.path.join(trn_path, MODEL_WEIGHTS_FILENAME)
    recovery_file = os.path.join(trn_path, RECOVER_TRAINING_FILENAME)

    if not os.path.exists(trn_path):
        os.makedirs(trn_path)

    trained = os.path.exists(train_model_state)
    recovery = os.path.exists(recovery_file)
    if trained and recovery:
        raise IOError('Both model state file and recovery state found inside training folder')

    if os.path.exists(os.path.join(trn_path, MODEL_HISTORY_FILENAME)):
        if not trained and not recovery:
            raise IOError('Training history found but model state information is missing')
    else:
        if trained or recovery:
            raise IOError('Model state found but training history is missing')

    last_init = 0
    if recovery:
        if trained:
            raise IOError('Recovery file found but model state information already exists')
        else:
            recovery_file = glob.glob(os.path.join(trn_path, RECOVER_TRAINING_FILENAME + '_*'))
            if len(recovery_file) > 0:
                last_init = int(recovery_file[0][:-1])

    model_info = os.path.exists(os.path.join(model_path, 'topology.json'))
    if not model_info:
        if trained:
            warnings.warn('Model state found in %s but topology information is missing from model folder'
                          'Data possibly corrupted' % train_model_state)
        fp = open(os.path.join(model_path, 'topology.json'), 'wb')
        json.dump(model_p.to_json(), fp, indent=1)
        fp.close()

    train_info = os.path.exists(os.path.join(trn_path, 'train_info.json'))
    if not train_info:
        if trained:
            warnings.warn('Model state found in %s but training information is missing from training folder'
                          'Data possibly corrupted' % train_model_state)
        fp = open(os.path.join(trn_path, 'train_info.json'), 'wb')
        json.dump(trn_p.to_json(), fp, indent=1)
        fp.close()
    else:
        if not trained:
            warnings.warn('Training information found but model state file is missing from training folder'
                          'This may happen if a training session is interrupted')

    return trn_path, last_init, trained

class NNParams(object):
    """
        Params Storage and builder Obj for Neural Network Based Classifier
    """

    def __init__(self,
                 optimizer=None,
                 layers=None,
                 loss=None,
                 metrics=None,
                 input_shape=(40, 400, 1)
                 ):

        # Default parameter settings -------------------------------
        if optimizer is None:
            optimizer = {'type': "SGD",
                         'lr': 0.01,
                         'decay': 1e-6,
                         'momentum': 0.9,
                         'nesterov': True}

        if loss is None:
            loss = "mean_squared_error"

        if metrics is None:
            metrics = ['acc', sp_index]

        # Setting parameters --------------------------------------
        self.__dict__ = OrderedDict()
        self.__dict__['input_shape'] = input_shape
        print(optimizer)
        if isinstance(optimizer, Optimizer):
            self.__dict__['optimizer'] = optimizer
        elif isinstance(optimizer, dict):
            self.__dict__['optimizer'] = Optimizer(optimizer)
        else:
            raise NotImplementedError("Implement error check")

        self.__dict__['layers'] = Layers()

        if layers is not None:
            for layer in layers:
                self.layers.add(layer)

        self.__dict__['loss'] = loss[0] if isinstance(loss, list) else loss
        self.__dict__['metrics'] = metrics

    @classmethod
    def from_json_file(cls, filepath):
        """
            Build parameters from parameters stored in json format
        :param filepath: path to file
        :return: NNParams object
        """
        params = joblib.load(filepath)
        return cls(*params)

    # TODO check reconstruction. Possible conflit with tuple and list
    def get_param_path(self):
        """
            Return path string given the object parameters
        :return: path string
        """
        path = self.optimizer.get_path_str() + '_'
        for layer in self.layers:
            path = path + layer.get_path_str()
        path = path[:-1] + '_' + self.loss
        for metric in self.metrics:
            str_metric = metric if isinstance(metric, str) else metric.__name__
            path = path + '_' + str_metric
        # path = path + '_' + str(self.epochs)
        # path = path + '_' + str(self.batch_size)

        return path

    def get_param_hash(self):
        """
            Calculate hash from the object parameters
        :return: Hash string
        """
        path = self.to_json()
        return hashlib.sha512(str(path)).hexdigest()

    def to_json(self):
        """
            Get parameters in json format
        :return: dictionary mapping the parameter names with its values
        """
        return {key: value if not (isinstance(value, Parameter) or isinstance(value, _ParameterSet))
                else value.to_json() for key, value in self}

    def __getitem__(self, param):
        return self.__dict__[param]

    def __getattr__(self, param):
        return self.__dict__[param]

    def __iter__(self):
        return self.next()

    def next(self):
        for param in self.__dict__:
            yield (param, self.__dict__[param])
        raise StopIteration


class Parameter(object):
    """
        NN parameter base class
    """

    def __init__(self, identifier, kwargs):
        self.__name__ = identifier

        # DEPRECATED identifier, use __name__
        self.__dict__['identifier'] = identifier
        self.__dict__['parameters'] = kwargs

    def __getitem__(self, param):
        return self.__dict__[param]

    def __getattr__(self, param):
        return self.__dict__[param]

    def __iter__(self):
        return self.next()

    def next(self):
        for param in self.__dict__:
            yield (param, self.__dict__[param])
        raise StopIteration

    def to_json(self):
        """

        :return:
        """
        return {'name': self.__name__,
                'args': self.parameters}

    def to_np_array(self):
        """

        :return:
        """
        return [self.__name__, self.parameters]

    def _to_keras_fn(self, keras_module):
        fn = getattr(keras_module, self.__name__)
        return fn(**self.parameters)


class _ParameterSet(object):
    def __init__(self):
        self.elements = list()

    def _add_element(self, instance_type, params):
        self.elements.append(instance_type(params))

    def _add(self, instance_type, element):

        if isinstance(element, instance_type):  # Case where Parameter type object is received
            self.elements.append(element)
        elif isinstance(element, dict):  # Case where parameters are passed in json format
            self._add_element(instance_type, element)  # args[0]['type'] contains Parameter name
        else:
            raise ValueError('received parameters must be a instance of %s or a dictionary containing'
                             'the name and arguments of the Parameter. '
                             '%s of type %s was passed' % (type(instance_type), element, type(element)))

    def __getitem__(self, i):
        return self.elements[i]

    def __iter__(self):
        return self.next()

    def __len__(self):
        return len(self.elements)

    def next(self):
        for element in self.elements:
            yield element
        raise StopIteration

    def to_json(self):
        """

        :return:
        """
        return {i: element.to_json() for i, element in enumerate(self.elements)}

    def to_np_array(self):
        """

        :return:
        """
        return [element.to_np_array() for element in self.elements]

    def to_keras_fn(self):
        """

        :return:
        """
        return [element.to_keras_fn() for element in self.elements]


class Optimizer(Parameter):
    """
        Optimizer class
    """
    method_list = ['SGD', 'Adam', 'Adagrad']

    def __init__(self, kwargs):
        method = kwargs["type"]
        del kwargs["type"]

        if method not in self.method_list:
            warnings.warn('Selected optimization method not found in method list.')

        super(Optimizer, self).__init__(method, kwargs)

    def to_keras_fn(self):
        """

        :return:
        """
        return self._to_keras_fn(keras.optimizers)

    def get_path_str(self):
        """

        :return:
        """
        return self.identifier


class Layer(Parameter):
    """
        Layer class
    """
    type_list = ['Conv2D', 'Conv1D', 'MaxPooling2D', 'MaxPooling1D', 'AveragePooling1D',
                 'AveragePooling2D', 'Flatten', 'Dense', 'Activation', 'Dropout']

    def __init__(self, kwargs):
        layer_type = kwargs["type"]
        del kwargs["type"]

        if layer_type not in self.type_list:
            warnings.warn('Selected layer type not found in layer types list')

        super(Layer, self).__init__(layer_type, kwargs)

    def to_keras_fn(self):
        """

        :return:
        """
        return self._to_keras_fn(keras.layers)

    def get_path_str(self):
        """

        :return:
        """
        param_str = self.__name__ + '_'
        for arg_name in self.parameters:
            if isinstance(self.parameters[arg_name], list) or isinstance(self.parameters[arg_name], tuple):
                param_str = param_str + arg_name + '_'
                for element in self.parameters[arg_name]:
                    param_str = param_str + str(element) + '_'
                param_str = param_str  # + '_'
            else:
                param_str = param_str + arg_name + '_' + str(self.parameters[arg_name]) + '_'
        return param_str


class Callback(Parameter):
    """
        Callback class
    """
    type_list = ['ModelCheckpoint', 'EarlyStopping', 'ReduceLROnPlateau', 'CSVLogger']

    def __init__(self, kwargs):
        callback = kwargs["type"]
        del kwargs["type"]

        if callback not in self.type_list:
            warnings.warn('Selected callback not found in callbacks list')

        super(Callback, self).__init__(callback, kwargs)

    def to_keras_fn(self):
        """

        :return:
        """
        return super(Callback, self)._to_keras_fn(keras.callbacks)

    def to_json(self):
        return {'name': self.__name__,
                'args': self.parameters}

class Layers(_ParameterSet):
    """
        Layer set class
    """

    def __init__(self, layers=None):
        super(Layers, self).__init__()

        if layers is not None:
            if isinstance(layers, list):
                for layer in layers:
                    if isinstance(layer, list):
                        new_layer = layer[1]
                        new_layer["type"] = layer[0]
                        layer = Layer(new_layer)
                    # elif layer is None:
                    #    layer = Layer()
                    self.add(layer)
            else:
                raise ValueError('layers must be a instance of list'
                                 '%s of type %s was passed' % (layers, type(layers)))

    def add(self, *args):
        """

        :param args:
        """
        self._add(Layer, *args)


class Callbacks(_ParameterSet):
    """
        Callbacks class
    """

    def __init__(self, callbacks=None):
        super(Callbacks, self).__init__()

        if callbacks is not None:
            if isinstance(callbacks, list):
                for callback in callbacks:
                    if isinstance(callback, list):
                        callback[1]["type"] = callback[0]
                        callback = Callback(**callback[1])
                    # elif callback is None:
                    #     callback = Callback()
                    self.add(callback)
            else:
                raise ValueError('callbacks must be a instance of list'
                                 '%s of type %s was passed' % (callbacks, type(callbacks)))

    def to_json(self):
        return {i:element.to_json() for i, element in enumerate(self.elements)}
    def add(self, *args):
        """

        :param args:
        """
        self._add(Callback, *args)
