import os
import warnings
from collections import OrderedDict

import joblib
import keras
from keras import Sequential, Model
from keras.engine.saving import load_model
from keras.utils import get_custom_objects

from lps_toolbox.metrics.classification import sp_index


class SequentialModelWrapper():
    """Keras Sequential Model wrapper class"""

    def __init__(self, trnParams, results_path):
        self._mountParams(trnParams)
        if not os.path.exists(results_path + '/' + trnParams.get_param_path()):
            self.model_path = os.path.join(results_path, trnParams.get_param_path())
            self.createFolders(self.model_path)
            self.saveParams(trnParams)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def _mountParams(self, trnParams):
        """Parses the parameters to attributes of the instance

           trnParams (TrnParamsConvolutional): parameter object
        """
        for param, value in trnParams:
            if param is None:
                raise ValueError('The parameters configuration received by SequentialModelWrapper must be complete'
                                 '%s was passed as NoneType.'
                                 % (param))
            setattr(self, param, value)

    def createFolders(self, *args):
        """Creates model folder structure from hyperparameters information"""
        for folder in args:
            os.makedirs(folder)

    def saveParams(self, trnParams):
        """Save parameters into a pickle file"""
        joblib.dump(trnParams.to_np_array(), os.path.join(self.model_path, 'model_info.jbl'))

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def __call__(self, *args, **kwargs):
        self.__init__(*args)

    def build_model(self):
        """Compile Keras model using the parameters loaded into the instance"""
        self.model = Sequential()
        for layer in self.layers:
            self.model.add(layer.to_keras_fn())
        print self.loss
        self.model.compile(optimizer=self.optimizer.to_keras_fn(),
                           loss=self.loss,
                           metrics=self.metrics)

    def fit(self, *args, **kwargs):
        # for arg in args:
        #     print arg.shape
        return self.model.fit(*args, **kwargs)

    def evaluate(self, x_test, y_test, verbose=0):
        """Model evaluation on a test set

            x_test: input data
            y_test: data labels

            :returns : evaluation results
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')

        test_results = self.model.evaluate(x_test,
                                           y_test,
                                           batch_size=self.batch_size,
                                           verbose=verbose)
        self.val_history = test_results
        return test_results

    def get_layer_n_output(self, n, data):
        """Returns the output of layer n of the model for the given data"""

        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.layers[n].output)
        intermediate_output = intermediate_layer_model.predict(data)
        return intermediate_output

    def predict(self, data, verbose=0):
        """Model evaluation on a data set

            :returns : model predictions (numpy.nparray / shape: <n_samples, n_outputs>
        """

        if self.model is None:
            raise StandardError('Model is not built. Run build method or load model before fitting')
        # print len(data[0])
        # print type(data[0])
        # print data.shape
        return self.model.predict(data, 1, verbose)  # ,steps)

    def load(self, file_path):
        """Load model info and weights from a .h5 file"""
        self.model = load_model(file_path)

    def save(self, file_path):
        """Save current model state and weights on an .h5 file"""
        self.model.save(file_path)

    def purge(self):
        """Delete model state"""
        del self.model



class CNNParams(object):
    def __init__(self,
                 prefix=None,
                 optimizer= None,
                 layers= None,
                 loss=None,
                 metrics=None,
                 epochs=10,
                 batch_size=32,
                 callbacks=None,
                 input_shape=(40, 400, 1)
                 ):

        # Default parameter settings -------------------------------
        if optimizer is None:
            optimizer = {"SGD": {'lr': 0.01,
                                 'decay': 1e-6,
                                 'momentum': 0.9,
                                 'nesterov': True}}

        if loss is None:
            loss = "mean_squared_error"

        if metrics is None:
            metrics = ['acc', sp_index]


        # Place input shape on first layer parameter list
        #layers[0][1]['input_shape'] = input_shape

        # Setting parameters --------------------------------------
        self.__dict__ = OrderedDict()
        self.__dict__['input_shape'] = input_shape
        self.__dict__['prefix'] = prefix

        self.__dict__['optimizer'] = Optimizer(optimizer)

        self.__dict__['layers'] = Layers()

        if layers is not None:
            for layer in layers:
                # name = layer['type']
                # args = layer
                # del args['type']
                self.layers.add(layer)

        self.__dict__['callbacks'] = callbacks
        if not callbacks is None:
            for args in callbacks:
                self.callbacks.add(args)

        self.__dict__['loss'] = loss[0] if isinstance(loss, list) else loss
        self.__dict__['metrics'] = metrics
        self.__dict__['epochs'] = epochs
        self.__dict__['batch_size'] = batch_size

    def toJson(self):
        return {key: value if not (isinstance(value, Parameter) or isinstance(value, _ParameterSet))
                else value.to_json() for key, value in self}

    @classmethod
    def fromfile(cls, filepath):
        params = joblib.load(filepath + '/model_info.jbl')
        return cls(*params)

    def __getitem__(self, param):
        return self.__dict__[param]

    def __getattr__(self, param):
        return self.__dict__[param]

    def __iter__(self):
        return self.next()

    def _storeParamObj(self, param, param_type, param_key, error_str):
        raise NotImplementedError

    def next(self):
        for param in self.__dict__:
            yield (param, self.__dict__[param])
        raise StopIteration

    def getParamPath(self):
        path = self.prefix + '/' + self.optimizer.get_path_str() + '/'
        for layer in self.layers:
            if layer.identifier == 'Activation':
                path = path + layer.parameters['activation'] + '_'
            else:
                path = path + layer.identifier + '_'
        path = path[:-1] + '/'  # remove last '_'
        for layer in self.layers:
            path = path + layer.get_path_str()
        path = path[:-1] + '/' + self.loss
        for metric in self.metrics:
            str_metric = metric if isinstance(metric, str) else metric.__name__
            path = path + '_' + str_metric
        path = path + '_' + str(self.epochs)
        path = path + '_' + str(self.batch_size)

        import hashlib

        hash = hashlib.sha512(path).hexdigest()

        #return path
        return hash + '_%i' % self.input_shape[0]

    def toNpArray(self):
        #print type(self.callbacks)
        if self.callbacks is not None:
            cbks = self.callbacks.to_np_array() # Compatibility purposes
        else:
            cbks = None

        return [self.prefix,
                self.optimizer.to_np_array(),
                self.layers.to_np_array(),
                self.loss,
                self.metrics,
                self.epochs,
                self.batch_size,
                cbks,
#                self.scale,
                self.layers.to_np_array()[0][1]['input_shape']
                ]

class Parameter(object):
    def __init__(self, identifier, kwargs):
        self.__dict__ = dict()

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

    # def __eq__(self, param2):
    #     if self.__name__ == param2.__name__:
    #         for (arg1, value1), (arg2, value2) in zip(self.parameters, param2.parameters):
    #             if not arg1 == arg2 or not value1 == value2:
    #                 return False
    #         return True
    #     return False

    def toJson(self):
        return {'name': self.__name__,
                'args': self.parameters}

    def toNpArray(self):
        return [self.__name__, self.parameters]

    def _toKerasFn(self, keras_module, name, args):
        fn = getattr(keras_module, name)
        return fn(**args)


class _ParameterSet(object):
    def __init__(self):
        self.elements = list()

    def _add_many(self, instance_type, elements):
        # DEPRECATED
        if isinstance(elements, list):
            for item in elements:
                self._add_element(instance_type, item)
        #
        # for value in elements:
        #     self._add_element(instance_type, value)

    def _add_element(self, instance_type, params):
        self.elements.append(instance_type(params))

    def _add(self, instance_type, *args):

        if len(args) == 1:
            if isinstance(args[0], instance_type):
                self.elements.append(args[0])
            elif isinstance(args[0], dict):
                self._add_element(instance_type, args[0])
            elif isinstance(args[0], list): # DEPRECATED # TODO remove
                self._add_many(instance_type, args[0])
            else:
                raise NotImplementedError
                # raise ValueError('element must be a instance of Callback'
                #                  '%s of type %s was passed' % (element, type(element)))
        elif len(args) == 2:
            parameters = None
            name = None
            for arg in args:
                if isinstance(arg, dict):
                    parameters = arg
                elif isinstance(arg, str):
                    name = arg

            if name is None or parameters is None:
                raise NotImplementedError
            self._add_element(instance_type, parameters)
        else:
            raise NotImplementedError

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

    def toJson(self):
        return {i:element.to_json() for i, element in enumerate(self.elements)}

    def pprint(self):
        raise NotImplementedError

    def toNpArray(self):
        return [element.to_np_array() for element in self.elements]

    def toKerasFn(self):
        return [element.to_keras_fn() for element in self.elements]

    def fitMask(self, mask):
        if not isinstance(mask, _ParameterSet):
            raise ValueError('mask type must be an instance of _ParameterSet'
                             '%s of type %s was passed' % (mask, type(mask)))
        if not mask is None:
            for submask, element in zip(mask, self):
                if not element.fitMask(submask):
                    return False
        return True


class Optimizer(Parameter):
    method_list = ['SGD', 'Adam', 'Adagrad']

    def __init__(self, kwargs):
        method = kwargs["type"]
        del kwargs["type"]

        if not method in self.method_list:
            warnings.warn('Selected optmization method not found in method list.')

        super(Optimizer, self).__init__(method, kwargs)

    def toKerasFn(self):
        return self._toKerasFn(keras.optimizers, self.__name__, self.parameters)

    def getPathStr(self):
        return self.identifier


class Layer(Parameter):
    type_list = ['Conv2D', 'Conv1D', 'MaxPooling2D', 'MaxPooling1D', 'AveragePooling1D', 'AveragePooling2D', 'Flatten', 'Dense', 'Activation', 'Dropout']
    type2path = ['c2', 'c1', 'mp2', 'mp1', 'ap1', 'ap2', '', 'd', '', 'drop']

    def __init__(self, kwargs):
        layer_type = kwargs["type"]
        del kwargs["type"]

        if not layer_type in self.type_list:
            warnings.warn('Selected layer type not found in layer types list')

        super(Layer, self).__init__(layer_type, kwargs)

    def toKerasFn(self):
        return self._toKerasFn(keras.layers, self.__name__, self.parameters)

    def getPathStr(self):
        l_map = dict()
        for ident, abv in zip(self.type_list, self.type2path):
            l_map[ident] = abv

        param_str = l_map[self.identifier] + '_'
        for parameter in self.parameters:
            if isinstance(self.parameters[parameter], list) or isinstance(self.parameters[parameter], tuple):
                param_str = param_str + parameter[0] + '_'
                for element in self.parameters[parameter]:
                    param_str = param_str + str(element) + '_'
                param_str = param_str  # + '_'
            else:
                param_str = param_str + parameter[0] + '_' + str(self.parameters[parameter]) + '_'
        return param_str


class Callback(Parameter):
    type_list = ['ModelCheckpoint', 'EarlyStopping', 'ReduceLROnPlateau', 'CSVLogger']

    def __init__(self, kwargs):
        callback = kwargs["type"]
        del kwargs["type"]

        if not callback in self.type_list:
            warnings.warn('Selected callback not found in callbacks list')

        super(Callback, self).__init__(callback, kwargs)

    def toKerasFn(self):
        return super(Callback, self)._toKerasFn(keras.callbacks, self.identifier, self.parameters)


class Layers(_ParameterSet):
    def __init__(self, layers=None):
        super(Layers, self).__init__()

        if not layers is None:
            if isinstance(layers, list):
                for layer in layers:
                    if isinstance(layer, list):
                        new_layer = layer[1]
                        new_layer["type"] = layer[0]
                        layer = Layer(new_layer)
                    elif layer == None:
                        layer = Layer()
                    self.add(layer)
            else:
                raise ValueError('layers must be a instance of list'
                                 '%s of type %s was passed' % (layers, type(layers)))

    def add(self, *args):
        self._add(Layer, *args)


class Callbacks(_ParameterSet):
    def __init__(self, callbacks=None):
        super(Callbacks, self).__init__()

        if not callbacks is None:
            if isinstance(callbacks, list):
                for callback in callbacks:
                    if isinstance(callback, list):
                        callback[1]["type"] = callback[0]
                        callback = Callback(**callback[1])
                    elif callback == None:
                        callback = Callback()
                    self.add(callback)
            else:
                raise ValueError('callbacks must be a instance of list'
                                 '%s of type %s was passed' % (callbacks, type(callbacks)))

    def add(self, *args):
        self._add(Callback, *args)