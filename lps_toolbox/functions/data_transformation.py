""" Implements generators for LOFAR images.
Authors :
Pedro Henrique Braga Lisboa 
Lucas Barra de Aguiar Nunes 
"""
import contextlib
import gc
import os
import warnings
import wave
from collections import OrderedDict

import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_memory

from lps_toolbox.functions.SystemIO import listfolders, listfiles, load, exists, save


class SonarRunsInfo():
    def __init__(self, inputdatapath, window, overlap=0, decimation_rate=1, verbose=False):
        self.inputdatapath = inputdatapath
        self.runs = OrderedDict()
        self.runs_named = OrderedDict()
        self.verbose = verbose

        if decimation_rate<1:
            decimation_rate=1

        # Load runs from folders
        class_offset = 0
        self.class_folders = list(listfolders(self.inputdatapath))
        self.class_folders.sort()
        for class_folder in self.class_folders:
            run_files = listfiles(self.inputdatapath + '/' + class_folder)
            run_files = list(run_files)
            run_names = list(map(lambda x: str(x[:-4]), run_files)) # remove .wav file extension / added list
            run_paths = list(map(lambda x: self.inputdatapath + '/' + class_folder + '/' + x, run_files)) #added list
            run_names.sort()
            run_paths.sort()
            run_indices = list(self._iterClassIndices(run_paths, class_offset, window, overlap, decimation_rate))
            if self.verbose:
                offsets = list(map(lambda x: x[0], run_indices))
                lengths = list(map(len, run_indices))
                print(class_folder)
                print("\tLength\tOffset")
                for (i, length), offset in zip(enumerate(lengths), offsets):
                    print("Run %i:\t%i\t%i" % (i, length, offset))
                print("Total: \t%i\n" % (sum(lengths)))

            class_offset = class_offset + sum(map(len, run_indices))

            self.runs[class_folder] = run_indices
            self.runs_named[class_folder] = {filename: indices for filename, indices in zip(run_names, run_indices)}
            """for key, value in self.runs.items():
                print(key,value)"""

    def _iterClassIndices(self, runpaths, class_offset, window, overlap=0, decimation_rate=1):
        run_offset = 0
        for run in runpaths:
            run_indices = self._getRunIndices(run, class_offset + run_offset, window, overlap=overlap, decimation_rate=decimation_rate)
            run_offset += len(run_indices)
            yield run_indices

    def _getRunIndices(self, runfile, offset, window, overlap=0, decimation_rate=1):
        with contextlib.closing(wave.open(runfile, 'r')) as runfile:
            frames = runfile.getnframes()
            frames = frames//decimation_rate
            end_frame = (frames - overlap) / (window - overlap) + offset
            
        return range(offset, int(end_frame)) #added int(end_frame)


class LofarKfoldGenerator():
    def __init__(self, data, target, freq, runs_info, window_size, stride, folds):
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
        
        folds: int
            Number of folds to be made
        """

        self.data = data
        self.target = target
        self.freq = freq
        self.runs_info = runs_info
        self.window_size = window_size
        self.stride = stride
        self.folds = folds
        self.shape = (len(self), self.window_size, len(self.data[0]))
        self.x_test = None
        self.y_test = None
        self.x_fit = None
        self.y_fit = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

    def split(self, fold, shuffle = False, validation = False, percentage = None, verbose = False):
        """
        Splits the datato the desired sets

        Parameters:
        
        fold: int
            current fold to make the split

        shuffle: boolean
            If True shuffles the data set
        
        validation: boolean
            If true splits the fit data into validation and train data

        percentage: float from 0 to 1
            Percentage of the fit data that will be used as validation data during the fit of the model

        verbose: boolean
            If true gives output of the process
        """

        windows, trgts = self._get_windows()
        split = int((len(trgts))/self.folds)

        #Testing if the fold divides without rest
        if len(trgts)/self.folds != 0:
            plus_1 = True

        #Shuffling the data
        if shuffle:
            window_trgt = list(zip(windows, trgts))
            np.random.shuffle(window_trgt)

            windows = list()
            trgts = list()
            for win, t in window_trgt:
                windows.append(win)
                trgts.append(t)
        
        current_fold = 0
        start = 0
        stop = split

        #Splitting the folds
        while (current_fold <= (self.folds-1)):
            if verbose:
                print(f'Splitting fold {current_fold}')
            if (current_fold == self.folds) and plus_1:
                stop = stop +1
            if current_fold == fold:
                self.x_test = windows[start:stop]
                self.y_test = trgts[start:stop]
            else:
                if type(self.x_fit) == type(None):
                    self.x_fit = windows[start:stop]
                    self.y_fit = trgts[start:stop]
                else:
                    self.x_fit.extend(windows[start:stop])
                    self.y_fit.extend(trgts[start:stop])
            current_fold += 1
        
        if verbose:
            print('The fold was splitted')

        #Splitting the validation data
        if validation:
            self.validation_split(percentage, shuffle)
        
        print('The data was splitted')

    def validation_split(self, percentage, shuffle = False):
        """
        Splits the fit data into trainning data and split data for fitting in .fit_generator keras method

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
        Returns two numpy arrays with the full valid set (data,class)
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
        Returns two numpy arrays with the full train set (data, class)
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
            yield self.data(win), win_cls
        
    def __len__(self):
        """Returns the length of the full windowed data"""
        windows, targets = self._get_windows()
        return len(targets)

    def _get_windows(self):
        """
        Get the windows' range from the data information
        
        Returns
            list, list: data, target respectively
        """

        windows = list()
        target = list()
        for run_cls, runs_array in enumerate(self.runs_info.runs.values()):
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

    def _loop_index(iterable, index):
        length = len(iterable)
        multiple = index/length
        if multiple <= 1:
            return index
        else:
            actual_index = index - (int(multiple)*length)
            return actual_index

def lofar2image(data, target, window_size, stride, runs_info, verbose = False):
    img_data = list()
    trgt = np.array([])
    for runs_array in runs_info.runs.values():
        for run_range in runs_array:
            run = data[run_range]
            window_array = _get_window_array(window_size, 0, run.shape[0], stride)
            run_start = run_range[0]
            for window_range in window_array:
                img_data.append(run[window_range])
                trgt = np.append(trgt, target[run_start])
    if verbose:
        print(f'Windows of size {window_size} with stride {stride}')
        for i, c in enumerate(runs_info.runs.keys()):
            print(f'{c} with {len(trgt[trgt == i])} events')
        print(f'Total of {len(trgt)} events')
    return np.array(img_data), trgt

def _get_window_array(window_size, start, stop, stride):
    window_array = list()
    for i in range(start, stop, stride):
        if i+window_size > stop:
            #The end of the range overpasses the run window size
            break
        window_array.append(range(i, i+window_size))
    return window_array

