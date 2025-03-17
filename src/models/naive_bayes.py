from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


class NaiveBayesModel(BaseEstimator, ClassifierMixin):
    """
    Naive Bayes Classifier for categorical and/or numeric features.

    :param categorical_feature_mask: List of booleans defining which features are categorical. Features that are not categorical are considered as numeric. Length of the list must match number of features.
    :param var_smoothing: Portion of the max variance of all numeric features, that is added to variances for calculation stability.
    :param laplace_smoothing: Additive smoothing parameter for categorical features.
    """

    def __init__(self, categorical_feature_mask: list[bool], var_smoothing: float = 1e-09,
                 laplace_smoothing: float = 1.0):
        self.categorical_feature_mask = categorical_feature_mask
        self.var_smoothing = var_smoothing
        self.laplace_smoothing = laplace_smoothing
        self._check_var_smoothing()
        self._check_laplace_smoothing()

    def _check_laplace_smoothing(self) -> None:
        if self.laplace_smoothing < 0:
            raise ValueError(f"Parameter laplace_smoothing = {self.laplace_smoothing} should be > 0")

    def _check_var_smoothing(self) -> None:
        if self.var_smoothing < 0:
            raise ValueError(f"Parameter var_smoothing = {self.var_smoothing} should be > 0")

    def _check_match_categorical_feature_mask(self, X: pd.DataFrame) -> None:
        if len(self.categorical_feature_mask) != X.shape[1]:
            raise ValueError(f"Length of categorical_feature_mask = {len(self.categorical_feature_mask)} must match "
                             f"number of features of X = {X.shape[1]}")

    def fit(self, X, y=None) -> None:
        """
        Fit the Mixed Naive Bayes according to X, y
        """

        X, y = check_X_y(X, y, dtype=None)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self._check_match_categorical_feature_mask(X)

        # get indices of categorical & numeric features
        self.idx_categorical_features_: list[int] = X.columns[np.array(self.categorical_feature_mask)].to_list()
        self.idx_numeric_features_: list[int] = X.columns[~np.array(self.categorical_feature_mask)].to_list()

        # count how often each class occurs
        class_counts = y.value_counts()
        self.classes_ = class_counts.index.tolist()
        self.class_priors_ = class_counts / len(self.classes_)

        # split dataframe X, into multiple dataframes foreach class
        X_by_class: dict[any, pd.DataFrame] = {}
        for class_value in self.classes_:
            X_by_class[class_value] = X[y == class_value]

        # determine probabilities of categorical features (per class)
        # structure of dict [class_value, feature_index, feature_value, float]
        self.p_by_class_: dict[any, dict[int, dict[any, float]]] = defaultdict(lambda: defaultdict(dict))
        for feature_idx in self.idx_categorical_features_:
            feature_counts_per_class = pd.crosstab(X[feature_idx], y)
            for class_value in self.classes_:
                feature_counts = feature_counts_per_class[class_value]
                probabilities = (feature_counts + self.laplace_smoothing) / (
                        class_counts[class_value] + self.laplace_smoothing * len(self.idx_categorical_features_))
                self.p_by_class_[class_value][feature_idx] = probabilities.to_dict()

        # "If the ratio of data variance between dimensions is too small, it
        # will cause numerical errors. To address this, we artificially
        # boost the variance" - sklearn
        self.var_boost_ = self.var_smoothing * np.var(X[self.idx_numeric_features_]).max()

        # determine mean and std of numeric features (per class)
        # structure of dicts [class_value, feature_index, float]
        self.means_ = np.empty(shape=(len(self.classes_), len(self.idx_numeric_features_)))
        self.vars_ = np.empty(shape=(len(self.classes_), len(self.idx_numeric_features_)))
        for i, class_value in enumerate(self.classes_):
            self.means_[i] = np.mean(X_by_class[class_value][self.idx_numeric_features_], axis=0)
            self.vars_[i] = np.var(X_by_class[class_value][self.idx_numeric_features_], axis=0) + self.var_boost_

        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True

    def predict(self, X) -> pd.DataFrame:
        """
        Perform classification on X, using the Mixed Naive Bayes.

        :param X: The input samples
        :return: Predicted target values for X.
        """

        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, dtype=None, ensure_min_features=self.n_features_in_)
        X = pd.DataFrame(X)
        self._check_match_categorical_feature_mask(X)

        def predict_row(row):
            # Categorical Features:
            # TODO: improve performance of categorial liklihood computation
            likelihoods_categorical = np.empty(shape=(len(self.classes_), len(self.idx_categorical_features_)))
            for i, class_value in enumerate(self.classes_):
                for j, feature_idx in enumerate(self.idx_categorical_features_):
                    feature_value = row[feature_idx]
                    # if no probability is found, just return 1, this ignores this features probability
                    likelihoods_categorical[i][j] = self.p_by_class_.get(class_value, {}).get(feature_idx, {}).get(feature_value, 1)
            # Numeric Features:
            numeric_feature_values = list(row[self.idx_numeric_features_])
            likelihoods_numeric = norm.pdf(numeric_feature_values, loc=self.means_, scale=np.sqrt(self.vars_))
            # log-likelihoods of each class, for numeric stability, since probabilities get 0 if the number of features is high
            likelihoods = np.concatenate([likelihoods_categorical, likelihoods_numeric], axis=1)
            log_likelihoods = np.sum(np.log(likelihoods), axis=1) + np.log(self.class_priors_)
            best_class = self.classes_[log_likelihoods.argmax(axis=0)]
            return best_class

        y_predict = X.apply(lambda x: predict_row(x), axis=1)
        return y_predict
    
    
