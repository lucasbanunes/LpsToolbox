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


class LofarKfoldGenerator(Lofar2ImgGenerator):
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
        super(LofarKfoldGenerator, self).__init__(data, target, freq, runs_info, window_size, stride)
        self.folds = folds
        self._folded = False

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
            answer = 'y'
            if self._folded:
                warnings.warn('The data was already splitted and folded, shuffling will alterate the folds.')
                answer = scan('Do you want to continue? (Y/N)')
                answer = answer.lower()
            if answer == 'y':
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
        
        self._folded = True
        
        print('The data was splitted')