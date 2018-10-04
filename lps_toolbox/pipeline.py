"""
The :mod lpsToolbok:pipeline module extends utilities from sklearn.pipeline.
"""
from sklearn import clone
from sklearn.pipeline import Pipeline
from sklearn.externals import six
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils import Bunch
from sklearn.utils.validation import check_memory

def _fit_transform_one(transformer, X, y, weight, **fit_params):
    print X.shape
    print y.shape
    if hasattr(transformer, 'fit_transform'):
        res, yt = transformer.fit_transform(X, y, **fit_params)
    else:
        res, yt = transformer.fit(X, y, **fit_params).transform(X)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, yt, transformer
    return res * weight, yt, transformer

class ExtendedPipeline(Pipeline):
    """Extends Pipeline object to handle transformers that returns both X and y values
    from fit and transform methods
    For more information on the original object and its functionalities,
    acess :ref: http://scikit-learn.org/stable/modules/compose.html
    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    See also
    --------
    sklearn.pipeline.make_pipeline : convenience function for simplified
        pipeline construction.
    Examples
    --------
    >>> from sklearn import svm
    >>> from sklearn.datasets import samples_generator
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.feature_selection import f_regression
    >>> from sklearn.pipeline import Pipeline
    >>> # generate some data to play with
    >>> X, y = samples_generator.make_classification(
    ...     n_informative=5, n_redundant=0, random_state=42)
    >>> # ANOVA SVM-C
    >>> anova_filter = SelectKBest(f_regression, k=5)
    >>> clf = svm.SVC(kernel='linear')
    >>> anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
    >>> # You can set the parameters using the names issued
    >>> # For instance, fit using a k of 10 in the SelectKBest
    >>> # and a parameter 'C' of the svm
    >>> anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
    ...                      # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
    Pipeline(memory=None,
             steps=[('anova', SelectKBest(...)),
                    ('svc', SVC(...))])
    >>> prediction = anova_svm.predict(X)
    >>> anova_svm.score(X, y)                        # doctest: +ELLIPSIS
    0.83
    >>> # getting the selected features chosen by anova_filter
    >>> anova_svm.named_steps['anova'].get_support()
    ... # doctest: +NORMALIZE_WHITESPACE
    array([False, False,  True,  True, False, False, True,  True, False,
           True,  False,  True,  True, False, True,  False, True, True,
           False, False])
    >>> # Another way to get selected features chosen by anova_filter
    >>> anova_svm.named_steps.anova.get_support()
    ... # doctest: +NORMALIZE_WHITESPACE
    array([False, False,  True,  True, False, False, True,  True, False,
           True,  False,  True,  True, False, True,  False, True, True,
           False, False])
    """

    # BaseEstimator interface

    def __init__(self, steps, memory=None):
        self.steps = steps
        self._validate_steps()
        self.memory = memory

    def get_params(self, deep=True):
        """Get parameters for this estimator.
        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._get_params('steps', deep=deep)

    def set_params(self, **kwargs):
        """Set the parameters of this estimator.
        Valid parameter keys can be listed with ``get_params()``.
        Returns
        -------
        self
        """
        self._set_params('steps', **kwargs)
        return self

    def _validate_steps(self):
        names, estimators = zip(*self.steps)

        # validate names
        self._validate_names(names)

        # validate estimators
        transformers = estimators[:-1]
        estimator = estimators[-1]

        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement fit and transform."
                                " '%s' (type %s) doesn't" % (t, type(t)))

        # We allow last estimator to be None as an identity transformation
        if estimator is not None and not hasattr(estimator, "fit"):
            raise TypeError("Last step of Pipeline should implement fit. "
                            "'%s' (type %s) doesn't"
                            % (estimator, type(estimator)))

    @property
    def _estimator_type(self):
        return self.steps[-1][1]._estimator_type

    @property
    def named_steps(self):
        # Use Bunch object to improve autocomplete
        return Bunch(**dict(self.steps))

    @property
    def _final_estimator(self):
        return self.steps[-1][1]

    # Estimator interface

    def _fit(self, X, y=None, **fit_params):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()
        # Setup the memory
        memory = check_memory(self.memory)

        fit_transform_one_cached = memory.cache(_fit_transform_one)

        fit_params_steps = dict((name, {}) for name, step in self.steps
                                if step is not None)
        for pname, pval in six.iteritems(fit_params):
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval

        Xt = X
        yt = y
        for step_idx, (name, transformer) in enumerate(self.steps[:-1]):
            if transformer is None:
                pass
            else:
                if hasattr(memory, 'location'):
                    # joblib >= 0.12
                    if memory.location is None:
                        # we do not clone when caching is disabled to
                        # preserve backward compatibility
                        cloned_transformer = transformer
                    else:
                        cloned_transformer = clone(transformer)
                elif hasattr(memory, 'cachedir'):
                    # joblib < 0.11
                    if memory.cachedir is None:
                        # we do not clone when caching is disabled to
                        # preserve backward compatibility
                        cloned_transformer = transformer
                    else:
                        cloned_transformer = clone(transformer)
                else:
                    cloned_transformer = clone(transformer)
                # Fit or load from cache the current transfomer
                Xt, yt, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer, Xt, yt, None,
                    **fit_params_steps[name])
                # Replace the transformer of the step with the fitted
                # transformer. This is necessary when loading the transformer
                # from the cache.
                self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator is None:
            return Xt, yt, {}
        return Xt, yt, fit_params_steps[self.steps[-1][0]]

    def fit(self, X, y=None, **fit_params):
        """Fit the model
        Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        self : Pipeline
            This estimator
        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:
            self._final_estimator.fit(Xt, yt, **fit_params)
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit the model and transform with the final estimator
        Fits all the transforms one after the other and transforms the
        data, then uses fit_transform on transformed data with the final
        estimator.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Transformed samples
        """
        last_step = self._final_estimator
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        if hasattr(last_step, 'fit_transform'):
            return last_step.fit_transform(Xt, yt, **fit_params)
        elif last_step is None:
            return Xt, yt
        else:
            return last_step.fit(Xt, yt, **fit_params).transform(Xt, yt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, **predict_params):
        """Apply transforms to the data, and predict with the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        **predict_params : dict of string -> object
            Parameters to the ``predict`` called at the end of all
            transformations in the pipeline. Note that while this may be
            used to return uncertainties from some models with return_std
            or return_cov, uncertainties that are generated by the
            transformations in the pipeline are not propagated to the
            final estimator.
        Returns
        -------
        y_pred : array-like
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt, _ = transform.transform(Xt)
        return self.steps[-1][-1].predict(Xt, **predict_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def fit_predict(self, X, y=None, **fit_params):
        """Applies fit_predict of last step in pipeline after transforms.
        Applies fit_transforms of a pipeline to the data, followed by the
        fit_predict method of the final estimator in the pipeline. Valid
        only if the final estimator implements fit_predict.
        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of
            the pipeline.
        y : iterable, default=None
            Training targets. Must fulfill label requirements for all steps
            of the pipeline.
        **fit_params : dict of string -> object
            Parameters passed to the ``fit`` method of each step, where
            each parameter name is prefixed such that parameter ``p`` for step
            ``s`` has key ``s__p``.
        Returns
        -------
        y_pred : array-like
        """
        Xt, yt, fit_params = self._fit(X, y, **fit_params)
        return self.steps[-1][-1].fit_predict(Xt, yt, **fit_params)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_proba(self, X):
        """Apply transforms, and predict_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_proba : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_proba(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def decision_function(self, X):
        """Apply transforms, and decision_function of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].decision_function(Xt)

    @if_delegate_has_method(delegate='_final_estimator')
    def predict_log_proba(self, X):
        """Apply transforms, and predict_log_proba of the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        y_score : array-like, shape = [n_samples, n_classes]
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt = transform.transform(Xt)
        return self.steps[-1][-1].predict_log_proba(Xt)

    @property
    def transform(self):
        """Apply transforms, and transform with the final estimator
        This also works where final estimator is ``None``: all prior
        transformations are applied.
        Parameters
        ----------
        X : iterable
            Data to transform. Must fulfill input requirements of first step
            of the pipeline.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_transformed_features]
        """
        # _final_estimator is None or has transform, otherwise attribute error
        # XXX: Handling the None case means we can't use if_delegate_has_method
        if self._final_estimator is not None:
            self._final_estimator.transform
        return self._transform

    def _transform(self, X):
        Xt = X
        for name, transform in self.steps:
            if transform is not None:
                Xt, yt = transform.transform(Xt)
        return Xt, yt

    @property
    def inverse_transform(self):
        """Apply inverse transformations in reverse order
        All estimators in the pipeline must support ``inverse_transform``.
        Parameters
        ----------
        Xt : array-like, shape = [n_samples, n_transformed_features]
            Data samples, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. Must fulfill
            input requirements of last step of pipeline's
            ``inverse_transform`` method.
        Returns
        -------
        Xt : array-like, shape = [n_samples, n_features]
        """
        # raise AttributeError if necessary for hasattr behaviour
        # XXX: Handling the None case means we can't use if_delegate_has_method
        for name, transform in self.steps:
            if transform is not None:
                transform.inverse_transform
        return self._inverse_transform

    def _inverse_transform(self, X):
        Xt = X
        for name, transform in self.steps[::-1]:
            if transform is not None:
                Xt = transform.inverse_transform(Xt)
        return Xt

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None, sample_weight=None):
        """Apply transforms, and score with the final estimator
        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of first step
            of the pipeline.
        y : iterable, default=None
            Targets used for scoring. Must fulfill label requirements for all
            steps of the pipeline.
        sample_weight : array-like, default=None
            If not None, this argument is passed as ``sample_weight`` keyword
            argument to the ``score`` method of the final estimator.
        Returns
        -------
        score : float
        """
        Xt = X
        for name, transform in self.steps[:-1]:
            if transform is not None:
                Xt, yt = transform.transform(Xt)
        score_params = {}
        if sample_weight is not None:
            score_params['sample_weight'] = sample_weight
        return self.steps[-1][-1].score(Xt, yt, **score_params)

    @property
    def classes_(self):
        return self.steps[-1][-1].classes_

    @property
    def _pairwise(self):
        # check if first estimator expects pairwise input
        return getattr(self.steps[0][1], '_pairwise', False)
