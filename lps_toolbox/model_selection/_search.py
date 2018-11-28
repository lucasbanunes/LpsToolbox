import os
import time
import warnings
from itertools import product

from sklearn.model_selection._search import BaseSearchCV, ParameterGrid, _check_param_grid
from sklearn.pipeline import Pipeline

from sklearn.base import is_classifier, clone

from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils import Parallel, delayed
from sklearn.externals import six
from sklearn.utils.validation import indexable
from sklearn.metrics.scorer import _check_multimetric_scoring


class PersGridSearchCV(BaseSearchCV):
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=None, iid='warn', refit=True, cv='warn', verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise-deprecating',
                 return_train_score="warn",
                 cachedir='./', return_estimator=False,
                 client=None):
        super(PersGridSearchCV, self).__init__(
            estimator=estimator, scoring=scoring, fit_params=fit_params,
            n_jobs=n_jobs, iid=iid, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        self.cachedir = cachedir
        self.return_estimator = return_estimator
        self.client = client
        _check_param_grid(param_grid)

    def fit(self, X, y=None, groups=None, **fit_params):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of the estimator
        """

        if self.fit_params is not None:
            warnings.warn('"fit_params" as a constructor argument was '
                          'deprecated in version 0.19 and will be removed '
                          'in version 0.21. Pass fit parameters to the '
                          '"fit" method instead.', DeprecationWarning)
            if fit_params:
                warnings.warn('Ignoring fit_params passed as a constructor '
                              'argument in favor of keyword arguments to '
                              'the "fit" method.', RuntimeWarning)
            else:
                fit_params = self.fit_params
        estimator = self.estimator
        cv = check_cv(self.cv, y, classifier=is_classifier(estimator))

        scorers, self.multimetric_ = _check_multimetric_scoring(
            self.estimator, scoring=self.scoring)

        if self.multimetric_:
            if self.refit is not False and (
                    not isinstance(self.refit, six.string_types) or
                    # This will work for both dict / list (tuple)
                    self.refit not in scorers):
                raise ValueError("For multi-metric scoring, the parameter "
                                 "refit must be set to a scorer key "
                                 "to refit an estimator with the best "
                                 "parameter setting on the whole data and "
                                 "make the best_* attributes "
                                 "available for that metric. If this is not "
                                 "needed, refit should be set to False "
                                 "explicitly. %r was passed." % self.refit)
            else:
                refit_metric = self.refit
        else:
            refit_metric = 'score'

        X, y, groups = indexable(X, y, groups)
        n_splits = cv.get_n_splits(X, y, groups)

        base_estimator = clone(self.estimator)

        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                            pre_dispatch=self.pre_dispatch)

        fit_and_score_kwargs = dict(scorer=scorers,
                                    fit_params=fit_params,
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True,
                                    return_times=True,
                                    return_parameters=False,
                                    return_estimator=self.return_estimator,
                                    error_score=self.error_score,
                                    verbose=self.verbose)
        results_container = [{}]
        with parallel:
            all_candidate_params = []
            all_out = []
            all_estimators = []

            def evaluate_candidates(candidate_params):
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)

                def _fit_and_score_recv(i_fold, X, y, train, test, parameters):
                    current_estimator = clone(base_estimator)
                    if isinstance(current_estimator, Pipeline):
                        if hasattr(current_estimator._final_estimator, 'cachedir'):
                            current_estimator._final_estimator.cachedir = os.path.join(self.cachedir,
                                                                                       '%i_fold ' % i_fold)
                        else:
                            warnings.warn('Final estimator does not have recovery'
                                          ' or saving capabilities')
                    elif hasattr(current_estimator, 'cachedir'):
                        current_estimator.cachedir = os.path.join(self.cachedir, '%i_fold ' % i_fold)
                    else:
                        warnings.warn('Estimator does not have recovery '
                                      ' or saving capabilities')
                    print parameters
                    print i_fold


                    return delayed(_fit_and_score)(current_estimator,
                                                    X, y,
                                                    train=train, test=test,
                                                    parameters=parameters,
                                                    **fit_and_score_kwargs)

                list_split = list(enumerate(cv.split(X, y, groups)))
                if self.verbose > 0:
                    print("Fitting {0} folds for each of {1} candidates,"
                          " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits))

                # print list(candidate_params)
                # raise NotImplementedError
                if self.client is None:
                    out = parallel(_fit_and_score_recv(i_fold,
                                                       X, y,
                                                       train, test,
                                                       parameters)
                                   for (parameters, (i_fold, (train, test)))
                                   in product(candidate_params,
                                              list_split))
                else:
                    self.client[:].use_dill()

                    dview = self.client[:]
                    out = dview.map(lambda parameters, i_fold, train, test: _fit_and_score_recv(i_fold,
                                                                                                X, y,
                                                                                                train, test,
                                                                                                parameters),
                                   [(parameters, i_fold, train, test) for (parameters, (i_fold, (train, test)))
                                                                      in product(candidate_params,
                                                                                 list_split)])

                if self.return_estimator:
                    all_estimators.extend([out_set[-1] for out_set in out])
                    out = [out_set[:-1] for out_set in out]
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)


                # XXX: When we drop Python 2 support, we can use nonlocal
                # instead of results_container
                results_container[0] = self._format_results(
                    all_candidate_params, scorers, n_splits, all_out)
                return results_container[0]

            self._run_search(evaluate_candidates)

        results = results_container[0]

        # For multi-metric evaluation, store the best_index_, best_params_ and
        # best_score_ iff refit is one of the scorer names
        # In single metric evaluation, refit_metric is "score"
        if self.refit or not self.multimetric_:
            self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
            self.best_params_ = results["params"][self.best_index_]
            self.best_score_ = results["mean_test_%s" % refit_metric][
                self.best_index_]

        if self.refit:
            self.best_estimator_ = clone(base_estimator).set_params(
                **self.best_params_)
            refit_start_time = time.time()
            if y is not None:
                self.best_estimator_.fit(X, y, **fit_params)
            else:
                self.best_estimator_.fit(X, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers if self.multimetric_ else scorers['score']

        self.cv_results_ = results
        self.n_splits_ = n_splits

        if self.return_estimator:
            self.cv_estimators = all_estimators

        return self

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""

        evaluate_candidates(ParameterGrid(self.param_grid))