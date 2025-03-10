import os
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, log_loss
import numpy as np

# Import utility functions
from utils import set_seed, DEFAULT_SEED, set_log

# Set the random seed for reproducibility
set_seed()

# Configure logging
logger = set_log()


def train_nb(X_train, y_train, var_smoothing = 1e-9):
    """
    Train a Gaussian Naïve Bayes model and return the training loss.

    :param X_train: Training features (NumPy array or Tensor).
    :param y_train: Training labels (NumPy array or Tensor).
    :return: Trained model and training loss.
    """
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    # Compute negative log-likelihood (log loss)
    y_prob = model.predict_proba(X_train)
    losses = log_loss(y_train, y_prob)
    return model, losses


def train_nb_optuna(X_train, y_train, trial):
    """
    Optimize Naïve Bayes hyperparameters using Optuna.

    :param X_train: Training features (NumPy array).
    :param y_train: Training labels (NumPy array).
    :param trial: Optuna trial object for hyperparameter tuning.
    :return: Log loss (to minimize).
    """
    var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1e-1, log=True)
    
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_train)
    loss = log_loss(y_train, y_prob) 

    return loss  