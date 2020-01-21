""" Implements generators for LOFAR images.
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

from SystemIO import listfolders, listfiles, load, exists, save


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
    def __init__(self, data, target, runs_info, window_size, stride, batch_size):
        self.data = data
        self.target = target
        self.length = len(target)
        self.runs_info = runs_info
        self.window_size = window_size
        self.stride = stride
        self.batch_size = batch_size
        self.x_test = None
        self.y_test = None
        self.x_train = None
        self.y_train = None

    def split_train_test(self, validation_split, shuffle = False):
        windows, trgts = self._get_windows()
        split = int(len(trgts)*validation_split)

        if shuffle:
            window_trgt = list(zip(windows, trgts))
            np.random.shuffle(window_trgt)

        win_shuff = list()
        trgt_shuff = list()
        for w, t in window_trgt:
            win_shuff.append(w)
            trgt_shuff.append(t)

        self.x_test = np.array(win_shuff[:split])
        self.y_test = np.array(trgt_shuff[:split])
        self.x_train = np.array(win_shuff[split:])
        self.y_train = np.array(trgt_shuff[split:])

        print('The data was splitted')
    
    def train_generator(self):

        """Generates the train data on demand"""

        steps = int(self.length/self.batch_size)
        start = 0
        for stop in range(self.batch_size, steps*self.batch_size, self.batch_size):
            batch = list()
            for w in self.x_train[start:stop]:
                batch.append(self.data[w])
            yield (np.array(batch), self.y_train[start:stop])
            start = stop
        
    def test_generator(self):
        raise NotImplementedError
        

    def _get_windows(self):

        """Get the windows range from the data information"""

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
        return np.array(windows), np.array(target)


def lofar2image_v2(data, target, window_size, stride, runs_info, verbose = False):
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