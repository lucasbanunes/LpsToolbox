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
from copy import deepcopy

import keras
import numpy as np
from keras.utils import to_categorical
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_memory

from lps_toolbox.functions.base_generator import Lofar2ImgGenerator
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
            run_files = listfiles(os.path.join(self.inputdatapath, class_folder))
            run_files = list(run_files)
            run_names = list(map(lambda x: str(x[:-4]), run_files)) # remove .wav file extension / added list
            run_paths = list(map(lambda x: os.path.join(self.inputdatapath, class_folder, x), run_files)) #added list
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


class LofarKfoldGenerator(Lofar2ImgGenerator):
    def __init__(self, data, target, freq, runs_info, window_size, stride, folds, novelty = False, novelty_class = None):
        """
        Parameters:

        data: numpy array
            LOFAR data
        
        target: numpy array
            Known classification of each data
        
        runs_info: list
            List with the ranges of each recorded run where the first dimension is the class

        window_size: int
            Vertical size of the window
        
        stride: int
            Stride made by the sliding window that mounts the immages
        
        folds: int
            Number of folds to be made

        novelty: boolean
            True if part of the data will be treated as novelty

        novelty_class: int
            Class of data that will be treated as novelty
        """
        super(LofarKfoldGenerator, self).__init__(data, target, freq, runs_info, window_size, stride, novelty, novelty_class)
        self.folds = folds
        self._folded = False
        self.current_fold = None
        self.x_folds = None
        self.y_folds = None

    def split(self, test_fold, shuffle = False, validation = False, percentage = None, verbose = False):
        """
        Splits the data to the desired sets

        Parameters:
        
        test_fold: int
            fold to be treated as the test fold

        shuffle: boolean
            If True shuffles the data set
        
        validation: boolean
            If true splits the fit data into validation and train data

        percentage: float from 0 to 1
            Percentage of the fit data that will be used as validation data during the fit of the model

        verbose: boolean
            If true gives output of the process
        """

        if not self._folded:

            #Getting the windowed data
            windows, trgts = self._get_windows()
            
            #Shuffling the data
            if shuffle:
                window_trgt = list(zip(windows, trgts))
                np.random.shuffle(window_trgt)
                windows = list()
                trgts = list()
                for w, t in window_trgt:
                    windows.append(w)
                    trgts.append(t)
                shuffle = False #To avoid entering the next shuffle unecessarily

            windows = np.array(windows)
            trgts = np.array(trgts)

            #Dividing into novelty and known data
            if self.novelty:
                self.x_novelty = windows[trgts == self.novelty_class]
                self.y_novelty = np.full((len(self.x_novelty), ), self.novelty_class)
                windows = windows[trgts != self.novelty_class]
                trgts = trgts[trgts != self.novelty_class]

            #Creating the folds
            _split = int((len(trgts))/self.folds)

            #Folding the data
            start = 0
            stop = _split
            self.x_folds = list()
            self.y_folds = list()

            for fold in range(self.folds):

                if verbose:
                    print(f'Splitting fold {fold}')

                self.x_folds.append(windows[start:stop])
                self.y_folds.append(trgts[start:stop])

            if verbose:
                print(f'The data as folded into {self.folds} folds.')
        
        if verbose:
            print(f'Splitting to fold {test_fold}.')

        #Shuffling the data
        if shuffle:
            for fold in range(len(self.y_folds)):
                window_trgt = list(zip(self.x_folds[fold], self.y_folds[fold]))
                np.random.shuffle(window_trgt)
                temp_data = list()
                temp_target = list()
                for data, target in window_trgt:
                    temp_data.append(data)
                    temp_target.append(target)
                self.x_folds[fold] = temp_data
                self.y_folds[fold] = temp_target
        
        #Assining each fold to is correspoding class
        self.x_test = self.x_folds[test_fold]
        self.y_test = self.y_folds[test_fold]

        #Copying data for preserving the original and removing the test
        self.x_fit = deepcopy(self.x_folds)
        self.x_fit.pop(test_fold)
        self.y_fit = deepcopy(self.y_folds)
        self.y_fit.pop(test_fold)
        self.x_fit = list(np.concatenate(tuple(self.x_fit), axis=0))
        self.y_fit = list(np.concatenate(tuple(self.y_fit), axis=0))

        #Splitting the validation data
        if validation:
            self.validation_split(percentage, shuffle)
        
        self._folded = True
        self.current_fold = test_fold
        
        if verbose:
            print('The data was splitted')

        gc.collect()


class LofarLeave1OutGenerator(Lofar2ImgGenerator):
    def __init__(self, data, target, freq, runs_info, window_size, stride, novelty = False, novelty_class = None):
        """
        Parameters:

        data: numpy array
            LOFAR data
        
        target: numpy array
            Known classification of each data
        
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
        super(LofarLeave1OutGenerator, self).__init__(data, target, freq, runs_info, window_size, stride, novelty, novelty_class)
    
    def split(self, run_class, run, shuffle = False, validation = False, percentage = None):
        """
        Splits the data to the desired sets

        Parameters:
        
        cls: int
            Index of the class that will be left out
        
        run: int
            Index of the run from the class "cls" that will be left out

        shuffle: boolean
            If True shuffles the data set
        
        validation: boolean
            If true splits the fit data into validation and train data

        percentage: float from 0 to 1
            Percentage of the fit data that will be used as validation data during the fit of the model
        """

        #Copying to mantain the original data untouched
        known_runs = deepcopy(self.runs_info)
        test_run = known_runs[run_class].pop(run)

        #Windowing the novelty data if it exists and rearranging the class value if it is over the novelty class

        if self.novelty:
            if run_class == self.novelty_class:
                raise ValueError("The class of the run to be left out can't be the novelty class.")
            self.x_novelty, self.y_novelty = self._get_windows([self.runs_info[self.novelty_class]], self.novelty_class)
            known_runs.pop(self.novelty_class)
            known_classes = np.delete(self.classes, self.novelty_class)
        else:
            known_classes = self.classes

        #Splitting and windowing the known data into fit and test data
        self.x_fit, self.y_fit = self._get_windows(known_runs, known_classes)
        self.x_test, self.y_test = self._get_windows(np.array([[test_run]]), run_class)

        #Shuffling the data
        if shuffle:
            window_trgt = list(zip(self.x_fit, self.y_fit))
            np.random.shuffle(window_trgt)

            self.x_fit = list()
            self.y_fit = list()
            for win, t in window_trgt:
                self.x_fit.append(win)
                self.y_fit.append(t)

        #Splitting the validation data
        if validation:
            self.validation_split(percentage, shuffle)
        
        gc.collect() #Collecting the garbage
        
        print('The data was splitted')