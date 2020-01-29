"""Base class for gor generators for LOFAR images
Author: Lucas Barra de Aguiar Nunes"""

import warnings
import numpy as np
from tensorflow.keras.utils import to_categorical

class Lofar2ImgGenerator():
    def __init__(self, data, target, freq, runs_info, window_size, stride):
        """
        Parameters:

        data: numpy array
            LOFAR data
        
        target: numpy array
            Known classification of each data
        
        runs_info: SonarRunsInfo
            The class generated for the specific LOFAR data that was passed

        window_size: int
            Vertical size of the window
        
        stride: int
            Stride made by the sliding window that mounts the immages
        """

        self.data = data
        self.target = target
        self.freq = freq
        self.runs_info = runs_info
        self.window_size = window_size
        self.stride = stride

        #Attributes for the data
        self.x_test = None
        self.y_test = None
        self.x_fit = None
        self.y_fit = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

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

    def get_valid_set(self, categorical = False):
        """
        Returns a tuple with two numpy arrays with the full valid set (data,class)
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

        if categorical:
            y_valid = to_categorical(y_valid)

        return (x_valid, y_valid)

    def get_train_set(self, categorical = False):
        """
        Returns a tuple with two numpy arrays with the full train set (data, class)
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

        if categorical:
            y_train = to_categorical(y_train)

        return (x_train, y_train)   

    def get_test_set(self, categorical = False):
        """
        Returns two numpy arrays with the full test set (data, class)
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

        if categorical:
            y_test = to_categorical(y_test)

        return (x_test, y_test)

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

    def train_generator(self, batch_size, epochs, shuffle = False, categorical = False):
        """
        Generates the train data on demand

        Parameters:

        batch_size: int
            Size of the batch to be generated

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
        for stop in range(batch_size, len(self.x_train)*epochs, batch_size):
            batch = list()
            start_index = self._loop_index(self.y_train, start)
            stop_index = self._loop_index(self.y_train, stop)

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

            if categorical:
                target = to_categorical(target)
            yield (batch, target)
            start = stop
        
    def test_generator(self, categorical = False):
        """
        Yields each lofar image with its respective known classficiation
        """
        for win, win_cls, in zip(self.x_test, self.y_test):
            if categorical:
                win_cls = to_categorical(win_cls)
            yield self.data[win], win_cls
        
    def __len__(self):
        """Returns the length of the full windowed data"""
        windows, targets = self._get_windows()
        return len(targets)

    def _get_windows(self, runs_values = None, monoclass = False, run_class = None):
        """
        Get the windows' range from the data information

        Parameters

        runs_values: iterable    
            Iterable with the shape (class, [data.shape])
        
        monoclass: boolean
            True if the runs_values had only one class, therefor its shape is only (data.shape)
        
        Returns
            list, list: data, target respectively
        """

        windows = list()
        target = list()

        if not runs_values:
            runs_values = self.runs_info.runs.values()
        
        if monoclass:
            runs_array = runs_values
            for run in runs_array:
                start = run[0]
                stop = run[-1]
                for i in range(start, stop, self.stride):
                    if i+self.window_size > stop:
                        #The end of the range overpasses the run window size
                        break
                    windows.append(range(i, i+self.window_size))
                    target.append(run_class)
        else:
            for run_cls, runs_array in enumerate(runs_values):
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

    def _loop_index(self, iterable, index):
        length = len(iterable)
        multiple = index/length
        if multiple <= 1:
            return index
        else:
            actual_index = index - (int(multiple)*length)
            return actual_index