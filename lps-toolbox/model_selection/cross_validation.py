import os
import re
import warnings
import joblib

class NestedCV:
    def __init__(self, n_cvs, n_folds):
        self.n_cvs = n_cvs
        self.cv_dict = None
        self.n_folds = n_folds

    def split_all(self, data, trgt, cv_obj):
        self.cv_dict = {cv_i: list(cv_obj.split(data, trgt))
                        for cv_i in range(self.n_cvs)}
        return self.cv_dict

    def save(self, filepath, namemask='', index_pos='first'):
        if not index_pos in ['first', 'last']:
            warnings.warn('index position mode %s is unknown, '
                          'fallback to first.' % index_pos,
                          RuntimeWarning)
            index_pos = 'first'

        for cv_i, cv_config in self.cv_dict.items():
            filename = self._nameHandle(namemask, cv_i, index_pos)
            joblib.dump(self.cv_dict, os.path.join(filepath, filename))

    def load(self, path, namemask, index_pos):
        if not index_pos in ['first', 'last']:
            warnings.warn('index position mode %s is unknown, '
                          'fallback to first.' % index_pos,
                          RuntimeWarning)
            index_pos = 'first'

        def is_fold_config(x):
            return not re.search(namemask, x) is None

        self.cv_dict = {self._getCVIndex(cv_filename, index_pos): joblib.load(os.path.join(path, cv_filename))
                        for cv_filename in filter(is_fold_config, os.listdir(path))}

    def _nameHandle(self, namemask, i, index_pos):
        str_i = str(i).zfill(len(str(self.n_cvs - 1)) - 1)
        if index_pos == 'last':
            filename = namemask + '_%i.jbl' % str_i
        else:
            filename = '%i_' % str_i  + namemask + '.jbl'

        return filename

    def _getCVIndex(self, filename, index_pos):
        n_digits = len(str(self.n_cvs - 1))
        if index_pos == 'first':
            return filename[:n_digits]
        return filename[n_digits:]