"""Base class for generators of LOFAR images
Author: Lucas Barra de Aguiar Nunes"""

import warnings
import numpy as np
from copy import deepcopy
from tensorflow.keras.utils import to_categorical

class Lofar2ImgGenerator():
    def __init__(self, data, target, freq, runs_info, window_size, stride, novelty = False, novelty_class = None):
        """
        Parameters:

        data: numpy array
            LOFAR data
        
        target: numpy array
            Known classification of each data

        freq: numpy array
            Array with the values of frequency used
        
        runs_info: list
            List with the ranges of each recorded run where the first dimension is the class

        window_size: int
            Vertical size of the window
        
        stride: int
            Stride made by the sliding window that mounts the immages

        novelty: boolean
            True if part of the data will be treated as novelty

        novelty_class: int
            Class of data that will be treated as novelty
        """

        #Full data information
        self.data = data
        self.target = target
        self.freq = freq
        self.classes , self.event_count= np.unique(self.target, return_counts = True)

        #Novelty data information
        self.novelty = novelty
        self.novelty_class = novelty_class
        
        if novelty:
            runs_info = deepcopy(runs_info)
            self.novelty_runs_info = [runs_info.pop(self.novelty_class)]
        else:
            self.novelty_runs_info = None

        #Information of the runs
        self.runs_info = runs_info
                
        #Windowed data information
        self.window_size = window_size
        self.stride = stride
        self.window_shape = (self.window_size, len(freq), 1)

        #Attributes for the windowed data
        self.x_test = None
        self.y_test = None
        self.x_fit = None
        self.y_fit = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_novelty = None
        self.y_novelty = None

    def split(self):
        raise NotImplementedError

    def validation_split(self, percentage, shuffle = False):
        """
        Splits the fit data into trainning data and validation data for fitting in .fit_generator keras method

        Parameters
        
        percentage: float between 0 and 1
            Percentage of the fit data that will be used as validation data
        
        shuffle: boolean
            If True shuffles the data
        """

        split = int((len(self.x_fit))*percentage)

        #Shuffling the data
        if shuffle:
            window_trgt = list(zip(self.x_fit, self.y_fit))
            np.random.shuffle(window_trgt)

            self.x_fit = list()
            self.y_fit = list()
            for win, t in window_trgt:
                self.x_fit.append(win)
                self.y_fit.append(t)

        #Splitting
        self.x_valid = self.x_fit[:split]
        self.x_train = self.x_fit[split:]
        self.y_valid = self.y_fit[:split]
        self.y_train = self.y_fit[split:]

    def get_valid_set(self, categorical = False, novelty_format = False):
        """
        Returns two numpy arrays with the full valid set data,class

        Parameters

        categorical: boolean
            If true the targeted classification array is outputted in categorical format

        novelty_format: boolean
            If true the targeted classification array is outputted considering the given novelty class

        Returns:

        x_valid, y_valid: numpy array, numpy array
            The full valid set data,class
        """

        if (not self.x_valid) and (not self.y_valid):
            warnings.warn('The data was not splitted the valid set is empty')

        x_valid = list()
        y_valid = list()

        for win, win_cls in zip(self.x_valid, self.y_valid):
            x_valid.append(self.data[win])
            y_valid.append(win_cls)
        
        x_valid = np.array(x_valid)
        x_valid = x_valid.reshape(-1, self.window_size, len(self.freq), 1)
        y_valid = np.array(y_valid)

        if novelty_format:
            y_valid = np.where(y_valid>self.novelty_class, y_valid-1, y_valid)
            if categorical:
                y_valid = to_categorical(y_valid, (len(self.classes)-1))
        elif categorical:
            y_valid = to_categorical(y_valid)

        return x_valid, y_valid

    def get_train_set(self, categorical = False, novelty_format = False):
        """
        Returns two numpy arrays with the full train set data, class

        Parameters

        categorical: boolean
            If true the targeted classification array is outputted in categorical format
        
        novelty_format: boolean
            If true the targeted classification array is outputted considering the given novelty class

        Returns:

        x_train, y_train: numpy array, numpy array
            The full train set data,class
        """

        if (not self.x_train) and (not self.y_train):
            warnings.warn('The data was not splitted the train set is empty')
        
        x_train = list()
        y_train = list()

        for win, win_cls in zip(self.x_train, self.y_train):
            x_train.append(self.data[win])
            y_train.append(win_cls)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        if novelty_format:
            y_train = np.where(y_train>self.novelty_class, y_train-1, y_train)
            if categorical:
                y_train = to_categorical(y_train, (len(self.classes)-1))
        elif categorical:
            y_train = to_categorical(y_train)

        return x_train, y_train   

    def get_test_set(self, categorical = False, novelty_format = False):
        """
        Returns two numpy arrays with the full test set data, class

        Parameters

        categorical: boolean
            If true the targeted classification array is outputted in categorical format

        novelty_format: boolean
            If true the targeted classification array is outputted considering the given novelty class

        Returns:

        x_test, y_test: numpy array, numpy array
            The full test set data,class
        """

        if (not self.x_test) and (not self.y_test):
            warnings.warn('The data was not splitted the test set is empty.')

        x_test = list()
        y_test = list()
        
        for win, win_cls in zip (self.x_test, self.y_test):
            x_test.append(self.data[win])
            y_test.append(win_cls)
        
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        if novelty_format:
            y_test = np.where(y_test>self.novelty_class, y_test-1, y_test)
            if categorical:
                y_test = to_categorical(y_test, (len(self.classes)-1))
        elif categorical:
            y_test = to_categorical(y_test)

        return (x_test, y_test)

    def get_novelty_set(self):
        """Returns  two numpy arrays with the full novelty set data, class
        
        Returns:
        
        x_novelty, y_novelty: numpy array, numpy array
            The full novelty set data,class
        """

        if not self.novelty:
            raise NameError("There's no novelty data set, you can define it at the initialization of the generator")

        x_novelty = list()
        y_novelty = list()

        for win, win_cls in zip(self.x_novelty, self.y_novelty):
            x_novelty.append(self.data[win])
            y_novelty.append(win_cls)

        x_novelty = np.array(x_novelty)
        y_novelty = np.array(y_novelty)

        return x_novelty, y_novelty

    def get_steps(self, batch_size):
        """
        Returns the number of steps necessary for one epoch in .fit_generator keras method
        
        Parameters:

         batch_size: int
            Size of the batch to be generated

        Returns:
        
        steps:int
            number of steps to be made
        """

        if (not self.x_train) and (not self.y_train):
            warnings.warn('The train set is empty. The data must be splitted before getting the steps values, returning zero.')
            return 0

        return int(len(self.x_train)/batch_size) 

    def train_generator(self, batch_size, epochs, n_inits=1,  shuffle = False, categorical = False):
        """
        Generates the train data on demand

        Parameters:

        batch_size: int
            Size of the batch to be generated

        epochs: int
            Number of epochs during the training of the model  

        n_inits: int
            Number of initializations of the model
        
        shuffle: boolean
            If true shuffles the data before feeding it to the model

        categorical: boolean
            If true the targeted classification array is outputted in categorical format

        Yields:

        (x_train_batch, y_train_batch: numpy array, numpy array): tuple
            Batch generated from the full train data
        """

        if (type(self.x_train) == None) or (type(self.y_train) == None):
            raise RuntimeError('The data must be splitted before generating the sets')

        if shuffle:
            window_trgt = list(zip(self.x_train, self.y_train))
            np.random.shuffle(window_trgt)

            self.x_train = list()
            self.y_train = list()
            for win, t in window_trgt:
                self.x_train.append(win)
                self.y_train.append(t)

        start = 0
        for stop in range(batch_size, len(self.x_train)*epochs*n_inits, batch_size):
            batch = list()
            start_index = self._loop_index(self.y_train, start)
            stop_index = self._loop_index(self.y_train, stop)

            #The indexes select the edges of the list
            if stop_index < start_index:
                target = np.array(self.y_train[start_index:])
                target = np.concatenate((target, self.y_train[:stop_index]))
                for win in self.x_train[start_index:]:
                    batch.append(self.data[win])
                for win in self.x_train[:stop_index]:
                    batch.append(self.data[win])
            
            else:
                target = np.array(self.y_train[start_index:stop_index])
                for win in self.x_train[start_index:stop_index]:
                    batch.append(self.data[win])
            
            batch = np.array(batch)
            batch = batch.reshape(batch_size, self.window_size, len(self.freq), 1) 

            if self.novelty:
                target = np.where(target>self.novelty_class, target-1, target)
                if categorical:
                    target = to_categorical(target, (len(self.classes)-1))
            elif categorical:
                target = to_categorical(target, len(self.classes))

            yield (batch, target)
            start = stop
        
    def test_generator(self, categorical = False):
        """
        Yields each lofar image with its respective known classficiation

        Parameters

        categorical: boolean
            If true the targeted classification array is outputted in categorical format

        Yields:
       
        img: numpy_array
            The data to be fed
        
        img_cls: numpy array
            Correct classification of img
        """
        for win, win_cls, in zip(self.x_test, self.y_test):
            if self.novelty:
                if win_cls>self.novelty_class:
                    img_cls = win_cls - 1
                else:
                    img_cls = win_cls
                if categorical:
                    img_cls = to_categorical(win_cls, (len(self.classes)-1))
            
            img = self.data[win]

            yield img, img_cls
        
    def novelty_generator(self):
        """
        Yields each lofar image with its respective known classficiation

        Yields:
       
        img: numpy_array
            The novelty data
        
        img_cls: numpy array
            Correct classification of the img
        """

        for win in self.x_novelty:
            img = self.data[win]

            yield img, self.y_novelty

    def __len__(self):
        """Returns the length of the full windowed data"""
        windows, targets = self._get_windows()
        return len(targets)

    def _get_windows(self, runs_values = None, run_class = None):
        """
        Get the windows' range from the data information
        Default configuration windows the entire data excpet the novelty data if it exists

        Parameters

        runs_values: iterable    
            Iterable with the shape (class, [data.shape]), variable from which the smaller images will be assembled

        run_class: int or iterable
            Targeted classification of the data
        
        Returns
            list, list: data, target respectively
        """

        windows = list()
        target = list()

        if not runs_values:
            runs_values = self.runs_info
        
        if not run_class:
            if self.novelty:
                run_class = np.delete(self.classes, self.novelty_class)
            else:
                run_class = self.classes
        
        if type(run_class) == int: #There is only one class
            run_class = np.full((len(runs_values), ), run_class)
        for run_cls, runs_array in zip(run_class, runs_values):
            for run in runs_array:
                start = run[0]
                stop = run[-1]
                for i in range(start, stop, self.stride):
                    if i+self.window_size > stop:
                        #The end of the range overpasses the run window size
                        break
                    windows.append(range(i, i+self.window_size))
                    target.append(run_cls)
    
        return windows, target

    def _loop_index(self, indexable, index):
        """Returns a valid index to be used with slicing as a loop
        
        Parameters:

        indexable: indexable object
            Object used for ccalculating its correct index

        index: int
            Index of the loop

        Returns:

        actual_index: int
            Atual usable index of the given object
        """
        length = len(indexable)
        multiple = index/length
        if multiple <= 1:
            return index
        else:
            actual_index = index - (int(multiple)*length)
            return actual_index