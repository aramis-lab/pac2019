"""Utility tools."""
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

import matplotlib.pyplot as plt


def plot_results(y_trues, y_preds, marker='o', ms=5, fillstyle=None,
                 linestyle='None', output_file=None):
    """Plot the results for each fold.

    Parameters
    ----------
    y_trues : list of arrays
        List of arrays with the true ages.

    y_preds : list of arrays
        List of arrays with the predicted ages.

    marker : str (default = 'o')
        Marker style.

    ms : float (default = 5)
        Marker size.

    fillstyle : None or str (default = None)
        Fillstyle.

    linestyle : None or str (default = 'None')
        Linestyle.

    output_file : None or str (default = None)
        If str, the plot is saved with the given name.

    """
    n_folds = len(y_trues)
    plt.figure(figsize=(6, 6 * n_folds))

    for i, (y_true, y_pred) in enumerate(zip(y_trues, y_preds)):

        plt.subplot(n_folds, 1, i + 1)

        # Plot each point
        plt.plot(y_true, y_pred, marker=marker, ms=ms,
                 fillstyle=fillstyle, linestyle=linestyle, color='C0')

        # Plot the perfect line
        min_age = np.min(np.r_[y_true, y_pred])
        max_age = np.max(np.r_[y_true, y_pred])
        plt.plot([min_age, max_age], [min_age, max_age], color='C1')

        # Compute the MAE
        mae = mean_absolute_error(y_true, y_pred)
        r, _ = spearmanr(y_true, np.abs(y_true - y_pred))

        # Add a title
        plt.title("Fold {0}\nMAE={1:0.3f}  -  r={2:0.3f}"
                  .format(i + 1, mae, r), fontsize=16)
        plt.xlabel("True age", fontsize=12)
        plt.ylabel("Predicted age", fontsize=12)

    plt.subplots_adjust(hspace=0.45)

    if output_file is not None:
        plt.savefig(output_file)

    plt.show()


class GridSearchCVRBFKernel(BaseEstimator, RegressorMixin):
    """GridSearchCV adapted for precomputed RBF kernel.

    Parameters
    ----------
    estimator : estimator
        Regressor.

    param_grid : dict
        Grid of parameters.

    cv : None, int or KFold instance (default = None)
        Cross validation iterable.

    age_range : None or tuple (default = None)
        Age range. If tuple, it will be used to clip the predictions.

    Attributes
    ----------
    best_estimator_ : estimator
        Best estimator

    cv_results_ : DataFrame
        Dataframe with all the cross validation results.

    """

    def __init__(self, estimator, param_grid, cv=None, age_range=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.age_range = age_range

    def fit(self, X, y, train_index):
        """Find the best combination of values.

        Parameters
        ----------
        X : array
            Input values.

        y : array
            Target values.

        """
        if self.cv is None:
            kfold = KFold(n_splits=5, shuffle=True)
        elif isinstance(self.cv, (int, np.integer)):
            kfold = KFold(n_splits=self.cv, shuffle=True)
        elif isinstance(self.cv, KFold):
            kfold = self.cv
        else:
            raise ValueError(
                "'cv' must be None, an integer or a KFold instance "
                "(got {0})".format(self.cv)
            )

        self._train_index = train_index

        gamma_values = []
        C_values = []
        mae_val_values = []
        mean_mae_val_values = []

        y_train = y[train_index]
        for gamma in self.param_grid['gamma']:
            X_rbf = np.exp(-gamma * X)
            X_train = X_rbf[train_index[:, None], train_index]

            for C in self.param_grid['C']:
                self.estimator.set_params(C=C)
                mae_val_split = []
                for train_train_index, train_val_index in kfold.split(
                    X_train, y_train
                ):
                    X_train_train = X_train[train_train_index[:, None],
                                            train_train_index]
                    X_train_val = X_train[train_val_index[:, None],
                                          train_train_index]
                    y_train_train = y_train[train_train_index]
                    y_train_val = y_train[train_val_index]

                    self.estimator.fit(X_train_train, y_train_train)
                    y_pred = self.estimator.predict(X_train_val)
                    if self.age_range is not None:
                        y_pred = np.clip(y_pred, *self.age_range)
                    score = mean_absolute_error(y_train_val, y_pred)

                    mae_val_split.append(score)

                gamma_values.append(gamma)
                C_values.append(C)
                mae_val_values.append(mae_val_split)
                mean_mae_val_values.append(np.mean(mae_val_split))

        idx = np.argmin(mean_mae_val_values)
        best_C = C_values[idx]
        best_gamma = gamma_values[idx]
        self.best_params_ = {'C': best_C, 'gamma': best_gamma}

        C_values = np.asarray(C_values).reshape(-1, 1)
        gamma_values = np.asarray(gamma_values).reshape(-1, 1)
        mae_val_values = np.asarray(mae_val_values).reshape(
            -1, kfold.get_n_splits())
        mean_mae_val_values = np.asarray(mean_mae_val_values).reshape(-1, 1)

        cv_results = np.c_[C_values,
                           gamma_values,
                           np.round(mae_val_values, 4),
                           np.round(mean_mae_val_values, 4)]
        columns = ['C', 'gamma']
        columns += ['test_score_split{0}'.format(i)
                    for i in range(mae_val_values.shape[1])]
        columns += ['mean_test_score']
        cv_results = pd.DataFrame(cv_results, columns=columns)
        self.cv_results_ = cv_results

        self._X_rbf = np.exp(- best_gamma * X)
        self._y = y
        self.best_estimator_ = self.estimator
        self.best_estimator_.set_params(C=best_C)
        self.best_estimator_.fit(self._X_rbf[train_index[:, None],
                                             train_index], y_train)

    def predict(self, test_index):
        """Predict.

        Parameters
        ----------
        test_index : array
            Indices for the test set.

        Returns
        -------
        y_pred : array
            Predicted values.

        """
        X_test = self._X_rbf[test_index[:, None], self._train_index]
        y_pred = self.best_estimator_.predict(X_test)
        if self.age_range is not None:
            y_pred = np.clip(y_pred, *self.age_range)
        return y_pred

    def score(self, test_index):
        """Predict and compute the mean absolute error.

        Parameters
        ----------
        test_index : array
            Indices for the test set.

        Returns
        -------
        mae : float
            Mean absolute error on the test set.

        """
        y_pred = self.predict(test_index)
        mae = mean_absolute_error(self._y[test_index], y_pred)
        return mae


class GridSearchCVLinearKernel(BaseEstimator, RegressorMixin):
    """GridSearchCV adapted for precomputed linear kernel.

    Parameters
    ----------
    estimator : estimator
        Regressor.

    param_grid : dict
        Grid of parameters.

    cv : None, int or KFold instance (default = None)
        Cross validation iterable.

    age_range : None or tuple (default = None)
        Age range. If tuple, it will be used to clip the predictions.

    Attributes
    ----------
    best_estimator_ : estimator
        Best estimator

    cv_results_ : DataFrame
        Dataframe with all the cross validation results.

    """

    def __init__(self, estimator, param_grid, cv=None, age_range=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.age_range = age_range

    def fit(self, X, y, train_index):
        """Find the best combination of values.

        Parameters
        ----------
        X : array
            Input values.

        y : array
            Target values.

        """
        if self.cv is None:
            kfold = KFold(n_splits=5, shuffle=True)
        elif isinstance(self.cv, (int, np.integer)):
            kfold = KFold(n_splits=self.cv, shuffle=True)
        elif isinstance(self.cv, KFold):
            kfold = self.cv
        else:
            raise ValueError(
                "'cv' must be None, an integer or a KFold instance "
                "(got {0})".format(self.cv)
            )

        self._train_index = train_index

        C_values = []
        mae_val_values = []
        mean_mae_val_values = []

        y_train = y[train_index]
        X_train = X[train_index[:, None], train_index]

        for C in self.param_grid['C']:
            self.estimator.set_params(C=C)
            mae_val_split = []
            for train_train_index, train_val_index in kfold.split(
                X_train, y_train
            ):
                X_train_train = X_train[train_train_index[:, None],
                                        train_train_index]
                X_train_val = X_train[train_val_index[:, None],
                                      train_train_index]
                y_train_train = y_train[train_train_index]
                y_train_val = y_train[train_val_index]

                self.estimator.fit(X_train_train, y_train_train)
                y_pred = self.estimator.predict(X_train_val)
                if self.age_range is not None:
                    y_pred = np.clip(y_pred, *self.age_range)
                score = mean_absolute_error(y_train_val, y_pred)

                mae_val_split.append(score)

            C_values.append(C)
            mae_val_values.append(mae_val_split)
            mean_mae_val_values.append(np.mean(mae_val_split))

        idx = np.argmin(mean_mae_val_values)
        best_C = C_values[idx]
        self.best_params_ = {'C': best_C}

        C_values = np.asarray(C_values).reshape(-1, 1)
        mae_val_values = np.asarray(mae_val_values).reshape(
            -1, kfold.get_n_splits())
        mean_mae_val_values = np.asarray(mean_mae_val_values).reshape(-1, 1)

        cv_results = np.c_[C_values,
                           np.round(mae_val_values, 4),
                           np.round(mean_mae_val_values, 4)]
        columns = ['C']
        columns += ['test_score_split{0}'.format(i)
                    for i in range(mae_val_values.shape[1])]
        columns += ['mean_test_score']
        cv_results = pd.DataFrame(cv_results, columns=columns)
        self.cv_results_ = cv_results

        self._X = X
        self._y = y
        self.best_estimator_ = self.estimator
        self.best_estimator_.set_params(C=best_C)
        self.best_estimator_.fit(self._X[train_index[:, None], train_index],
                                 y_train)

    def predict(self, test_index):
        """Predict.

        Parameters
        ----------
        test_index : array
            Indices for the test set.

        Returns
        -------
        y_pred : array
            Predicted values.

        """
        X_test = self._X[test_index[:, None], self._train_index]
        y_pred = self.best_estimator_.predict(X_test)
        if self.age_range is not None:
            y_pred = np.clip(y_pred, *self.age_range)
        return y_pred

    def score(self, test_index):
        """Predict and compute the mean absolute error.

        Parameters
        ----------
        test_index : array
            Indices for the test set.

        Returns
        -------
        mae : float
            Mean absolute error on the test set.

        """
        y_pred = self.predict(test_index)
        mae = mean_absolute_error(self._y[test_index], y_pred)
        return mae
