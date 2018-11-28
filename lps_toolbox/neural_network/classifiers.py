"""Neural network based classifiers

Authors: Pedro Henrique Braga Lisboa <pedro.lisboa@lps.ufrj.br>
"""

import keras
import inspect
import numpy as np

from lps_toolbox.metrics.classification import sp_index, recall_score
from lps_toolbox.neural_network.base import BaseNNClassifier, Layers


class ConvNetClassifier(BaseNNClassifier):
    """
        Convolutional Network Classifier
        Compatible with sklearn interface

        Currently, this implementation only supports topologies where each convolutional layer
        is followed by a pooling layer. Future implementations will remove this limitation
    """
    # TODO remove limitation where each convolutional layer is followed by a pooling layer
    # Solution1: pool_filter_sizes must have None where the convolutional layers has no pooling
    # Solution2: insert some index information on pool_filter_sizes parameter
    #            ex.: pool_filter_sizes=((<conv_layer_idx>, <filter_size>))
    # Solution3: insert separated parameter with the index of the pooling layers. Defaults to None where
    #            the pooling layers will be interspersed
    def __init__(self,
                 input_shape=(None,),
                 n_filters=(6,),
                 conv_filter_sizes=((4, 4),),
                 conv_strides=((1, 1),),
                 pool_filter_sizes=((2, 2),),
                 pool_strides=((1, 1),),
                 conv_activations=("relu",),
                 conv_padding=('valid',),
                 conv_dropout=None,
                 pool_padding=('valid',),
                 pool_types=('MaxPooling',),
                 conv_dilation_rate=(1,),
                 pool_dilation_rate=(1,),
                 data_format='channels_last',
                 dense_layer_sizes=(10,),
                 dense_activations=("softmax",),
                 dense_dropout=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 solver="adam",
                 batch_size=32,
                 epochs=200,
                 loss="categorical_crossentropy",
                 metrics=None,
                 momentum=0.9,
                 nesterov=True,
                 decay=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 learning_rate=0.001,
                 amsgrad=False,
                 early_stopping=False,
                 es_kwargs=None,
                 model_checkpoint=True,
                 save_best=True,
                 mc_kwargs=None,
                 log_history=True,
                 cachedir='./'):
        """

        :param input_shape:
        :param n_filters:
        :param conv_filter_sizes:
        :param conv_strides:
        :param pool_filter_sizes:
        :param pool_strides:
        :param conv_activations:
        :param conv_padding:
        :param conv_dropout:
        :param pool_padding:
        :param pool_types:
        :param conv_dilation_rate:
        :param pool_dilation_rate:
        :param data_format:
        :param dense_layer_sizes:
        :param dense_activations:
        :param dense_dropout:
        :param kernel_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param activity_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
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
        super(ConvNetClassifier, self).__init__(input_shape, solver, batch_size, epochs, loss, metrics,
                                                momentum, nesterov, decay, beta_1, beta_2,
                                                epsilon, learning_rate, amsgrad, early_stopping, es_kwargs,
                                                model_checkpoint, save_best, mc_kwargs, log_history, cachedir)

        self.n_filters = n_filters
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_strides = conv_strides
        self.pool_filter_sizes = pool_filter_sizes
        self.pool_strides = pool_strides
        self.conv_activations = conv_activations
        self.conv_padding = conv_padding
        self.conv_dropout = conv_dropout
        self.pool_padding = pool_padding
        self.pool_types = pool_types
        self.conv_dilation_rate = conv_dilation_rate
        self.pool_dilation_rate = pool_dilation_rate
        self.data_format = data_format
        self.dense_layer_sizes = dense_layer_sizes
        self.dense_activations = dense_activations
        self.dense_dropout = dense_dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def _build_topology(self):
        # Building convolutional layers ---------------------------------------------------------------
        conv_layers_parameters = zip(self.n_filters, self.conv_filter_sizes, self.conv_strides,
                                     self.conv_activations, self.conv_padding, self.conv_dilation_rate)
        conv_layers = list()
        for filters, kernel_size, stride, activation, padding, dilation_rate in conv_layers_parameters:
            conv_layers.append(
                self.build_conv_layer(filters, kernel_size, stride, activation, padding,
                                      self.data_format, dilation_rate, self.kernel_initializer,
                                      self.bias_initializer, self.kernel_regularizer,
                                      self.bias_regularizer, self.activity_regularizer,
                                      self.kernel_constraint, self.bias_constraint)
            )
        # -----------------------------------------------------------------------------------------------

        # conv_layers = [self.build_conv_layer(filters,
        #                                      kernel_size,
        #                                      stride,
        #                                      activation,
        #                                      padding,
        #                                      self.data_format,
        #                                      dilation_rate,
        #                                      self.kernel_initializer,
        #                                      self.bias_initializer,
        #                                      self.kernel_regularizer,
        #                                      self.bias_regularizer,
        #                                      self.activity_regularizer,
        #                                      self.kernel_constraint,
        #                                      self.bias_constraint)
        #                for filters, kernel_size, stride, activation, padding, dilation_rate
        #                in zip(self.n_filters,
        #                       self.conv_filter_sizes,
        #                       self.conv_strides,
        #                       self.conv_activations,
        #                       self.conv_padding,
        #                       self.conv_dilation_rate)
        #                ]

        # Building pooling layers --------------------------------------------------------------
        pool_layers_parameters = zip(self.pool_types, self.pool_filter_sizes,
                                     self.pool_strides, self.pool_padding)
        pool_layers = list()
        for pool_type, pool_size, stride, padding in pool_layers_parameters:
            pool_layers.append(self.build_pooling_layer(pool_type, pool_size,
                                                        stride, padding, self.data_format))

        # --------------------------------------------------------------------------------------

        # pool_layers = [self.build_pooling_layer(pool_type,
        #                                         pool_size,
        #                                         stride,
        #                                         padding,
        #                                         self.data_format)
        #                for pool_type, pool_size, stride, padding
        #                in zip(self.pool_types,
        #                       self.pool_filter_sizes,
        #                       self.pool_strides,
        #                       self.pool_padding)
        #                ]

        def intersperse(iter1, iter2):
            for el1, el2 in zip(iter1, iter2):
                yield el1
                yield el2

        conv_pool_layers = list(intersperse(conv_layers, pool_layers))
        conv_pool_layers.append({"type": "Flatten"})

        # Building dense layers ----------------------------------------------------------
        dense_layers_parameters = zip(self.dense_layer_sizes, self.dense_activations)
        dense_layers = list()
        for units, activation in dense_layers_parameters:
            dense_layers.append(
                self.build_dense_layer(units, activation, self.kernel_initializer,
                                       self.bias_initializer, self.kernel_regularizer,
                                       self.bias_regularizer, self.activity_regularizer,
                                       self.kernel_constraint, self.bias_constraint)
            )
        # ---------------------------------------------------------------------------------

        # dense_layers = [self.build_dense_layer(units,
        #                                        activation,
        #                                        self.kernel_initializer,
        #                                        self.bias_initializer,
        #                                        self.kernel_regularizer,
        #                                        self.bias_regularizer,
        #                                        self.activity_regularizer,
        #                                        self.kernel_constraint,
        #                                        self.bias_constraint)
        #                 for units, activation in zip(self.dense_layer_sizes, self.dense_activations)]

        if self.dense_dropout is not None:
            for position, rate in enumerate(self.dense_dropout):
                dense_layers = dense_layers[:position] + [keras.layers.Dropout(rate)] + dense_layers[position:]

        layers = np.concatenate([conv_pool_layers, dense_layers])
        layers[0]["input_shape"] = self.input_shape

        layers_wrapper = Layers()
        if layers is not None:
            for layer in layers:
                layers_wrapper.add(layer)

        return layers_wrapper

    @staticmethod
    def build_conv_layer(filters,
                         kernel_size,
                         stride,
                         activation,
                         padding,
                         data_format,
                         dilation_rate,
                         kernel_initializer,
                         bias_initializer,
                         kernel_regularizer,
                         bias_regularizer,
                         activity_regularizer,
                         kernel_constraint,
                         bias_constraint,
                         input_shape=None):
        """

        :param filters:
        :param kernel_size:
        :param stride:
        :param activation:
        :param padding:
        :param data_format:
        :param dilation_rate:
        :param kernel_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param activity_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param input_shape:
        :return:
        """
        conv_dim = len(kernel_size)

        if len(stride) > conv_dim:
            raise NotImplementedError
        elif len(stride) < conv_dim and len(stride) != 1:
            raise NotImplementedError
        if not isinstance(filters, int):
            raise NotImplementedError
        if not isinstance(activation, str):
            raise NotImplementedError
        if not isinstance(padding, str):
            raise NotImplementedError
        if not isinstance(data_format, str):
            raise NotImplementedError

        if conv_dim == 1:
            layer_type = "Conv1D"
        elif conv_dim == 2:
            layer_type = "Conv2D"
        elif conv_dim == 3:
            layer_type = "Conv3D"
        else:
            raise NotImplementedError
        # TODO handle other types of convolutions

        layer = {"type": layer_type,
                 "filters": filters,
                 "kernel_size": kernel_size,
                 "strides": stride,
                 "padding": padding,
                 "data_format": data_format,
                 "dilation_rate": dilation_rate,
                 "activation": activation,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        return layer

    @staticmethod
    def build_pooling_layer(layer_type,
                            pool_size,
                            stride,
                            padding,
                            data_format):
        """

        :param layer_type: 
        :param pool_size: 
        :param stride: 
        :param padding: 
        :param data_format: 
        :return: 
        """
        pool_dim = len(pool_size)

        if len(stride) > pool_dim:
            raise NotImplementedError
        elif len(stride) < pool_dim and len(stride) != 1:
            raise NotImplementedError

        if not isinstance(padding, str):
            raise NotImplementedError

        # TODO handle other types of poolings
        if layer_type not in ["MaxPooling", "AveragePooling"]:
            raise NotImplementedError
        if pool_dim == 1:
            layer_type += "1D"
        elif pool_dim == 2:
            layer_type += "2D"
        elif pool_dim == 3:
            layer_type += "3D"

        layer = {"type": layer_type,
                 "pool_size": pool_size,
                 "strides": stride,
                 "padding": padding,
                 "data_format": data_format}

        return layer

    @staticmethod
    def build_dense_layer(units,
                          activation,
                          kernel_initializer,
                          bias_initializer,
                          kernel_regularizer,
                          bias_regularizer,
                          activity_regularizer,
                          kernel_constraint,
                          bias_constraint,
                          input_shape=None):
        """

        :param units:
        :param activation:
        :param kernel_initializer:
        :param bias_initializer:
        :param kernel_regularizer:
        :param bias_regularizer:
        :param activity_regularizer:
        :param kernel_constraint:
        :param bias_constraint:
        :param input_shape:
        :return:
        """
        layer = {"type": "Dense",
                 "units": units,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        if input_shape is not None:
            layer["input_shape"] = input_shape

        if activation != "":
            layer["activation"] = activation
        return layer


class MLPClassifier(BaseNNClassifier):
    def __init__(self,
                 layer_sizes=(10,),
                 activations=("relu",),
                 solver="adam",
                 batch_size=32,
                 epochs=200,
                 loss="categorical_crossentropy",
                 metrics=["acc"],
                 input_shape=(None,),
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 momentum=0.9,
                 nesterov=True,
                 decay=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-08,
                 learning_rate=0.001,
                 amsgrad=False,
                 early_stopping=False,
                 es_kwargs=None,
                 model_checkpoint=True,
                 save_best=True,
                 mc_kwargs=None,
                 log_history=True,
                 cachedir='./'):

        self.cachedir = cachedir
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _build_topology(self):
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

        layers = [self.build_layer(units,
                                   activation,
                                   self.kernel_initializer,
                                   self.bias_initializer,
                                   self.kernel_regularizer,
                                   self.bias_regularizer,
                                   self.activity_regularizer,
                                   self.kernel_constraint,
                                   self.bias_constraint)
                  for units, activation in zip(self.layer_sizes, self.activations)]
        layers[0] = self.build_layer(self.layer_sizes[0],
                                     self.activations[0],
                                     self.kernel_initializer,
                                     self.bias_initializer,
                                     self.kernel_regularizer,
                                     self.bias_regularizer,
                                     self.activity_regularizer,
                                     self.kernel_constraint,
                                     self.bias_constraint,
                                     input_shape=self.input_shape)

        layers_obj = Layers()
        if layers is not None:
            for layer in layers:
                layers_obj.add(layer)

        return layers_obj

    def score(self, X, y, sample_weight=None, return_eff=True):
        if y.ndim > 1:
            y = y.argmax(axis=1)

        out = self.predict(X)

        cat_out = out.argmax(axis=1)

        if return_eff:
            recall = recall_score(y, cat_out)
            scores = dict()
            scores['eff_0'] = recall[0]
            scores['eff_1'] = recall[1]
            scores['eff_2'] = recall[2]
            scores['eff_3'] = recall[3]

            scores['sp'] = sp_index(y, cat_out)
            return scores

        return sp_index(y, cat_out)

    @staticmethod
    def build_layer(units,
                    activation,
                    kernel_initializer,
                    bias_initializer,
                    kernel_regularizer,
                    bias_regularizer,
                    activity_regularizer,
                    kernel_constraint,
                    bias_constraint,
                    input_shape=None):

        layer = {"type": "Dense",
                 "units": units,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        if input_shape is not None:
            layer["input_shape"] = input_shape

        if activation != "":
            layer["activation"] = activation
        return layer

    @staticmethod
    def build_conv_layer(filters,
                         kernel_size,
                         stride,
                         activation,
                         padding,
                         data_format,
                         dilation_rate,
                         kernel_initializer,
                         bias_initializer,
                         kernel_regularizer,
                         bias_regularizer,
                         activity_regularizer,
                         kernel_constraint,
                         bias_constraint,
                         input_shape=None):

        conv_dim = len(kernel_size)

        if len(stride) > conv_dim:
            raise NotImplementedError
        elif len(stride) < conv_dim and len(stride) != 1:
            raise NotImplementedError

        if not isinstance(filters, int):
            raise NotImplementedError
        if not isinstance(activation, str):
            raise NotImplementedError
        if not isinstance(padding, str):
            raise NotImplementedError
        if not isinstance(data_format, str):
            raise NotImplementedError

        if conv_dim == 1:
            type = "Conv1D"
        elif conv_dim == 2:
            type = "Conv2D"
        elif conv_dim == 3:
            type = "Conv3D"
        else:
            raise NotImplementedError
        # TODO handle other types of convolutions

        layer = {"type": type,
                 "filters": filters,
                 "kernel_size": kernel_size,
                 "strides": stride,
                 "padding": padding,
                 "data_format": data_format,
                 "dilation_rate": dilation_rate,
                 "activation": activation,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        return layer

    @staticmethod
    def build_pooling_layer(type,
                            pool_size,
                            stride,
                            padding,
                            data_format):

        pool_dim = len(pool_size)

        if len(stride) > pool_dim:
            raise NotImplementedError
        elif len(stride) < pool_dim and len(stride) != 1:
            raise NotImplementedError

        if not isinstance(padding, str):
            raise NotImplementedError

        # TODO handle other types of poolings
        if type not in ["MaxPooling", "AveragePooling"]:
            raise NotImplementedError
        if pool_dim == 1:
            type += "1D"
        elif pool_dim == 2:
            type += "2D"
        elif pool_dim == 3:
            type += "3D"

        layer = {"type": type,
                 "pool_size": pool_size,
                 "strides": stride,
                 "padding": padding,
                 "data_format": data_format}

        return layer

    @staticmethod
    def build_dense_layer(units,
                          activation,
                          kernel_initializer,
                          bias_initializer,
                          kernel_regularizer,
                          bias_regularizer,
                          activity_regularizer,
                          kernel_constraint,
                          bias_constraint,
                          input_shape=None):

        layer = {"type": "Dense",
                 "units": units,
                 "kernel_initializer": kernel_initializer,
                 "bias_initializer": bias_initializer,
                 "kernel_regularizer": kernel_regularizer,
                 "bias_regularizer": bias_regularizer,
                 "activity_regularizer": activity_regularizer,
                 "kernel_constraint": kernel_constraint,
                 "bias_constraint": bias_constraint}

        if input_shape is not None:
            layer["input_shape"] = input_shape

        if activation != "":
            layer["activation"] = activation
        return layer
