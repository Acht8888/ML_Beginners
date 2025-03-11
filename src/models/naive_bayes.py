import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from optuna.samplers import TPESampler

from utils import set_seed, DEFAULT_SEED

set_seed()

class NaiveBayesModel:
    def __init__(self):
        self.class_prior = None
        self.means = None
        self.variances = None
    
    def fit(self, X_train, y_train, var_smoothing=1e-9):
        classes = torch.unique(y_train)
        n_classes = len(classes)
        n_features = X_train.shape[1]
        
        self.class_prior = torch.zeros(n_classes)
        self.means = torch.zeros((n_classes, n_features))
        self.variances = torch.zeros((n_classes, n_features))
        
        for i, cls in enumerate(classes):
            X_cls = X_train[y_train == cls]
            self.class_prior[i] = len(X_cls) / len(X_train)
            self.means[i] = X_cls.mean(dim=0)
            self.variances[i] = X_cls.var(dim=0) + var_smoothing  # Add smoothing
    
    def predict(self, X_test):
        log_probabilities = []
        for i in range(len(self.class_prior)):
            prior = torch.log(self.class_prior[i])
            exponent = -0.5 * ((X_test - self.means[i]) ** 2) / self.variances[i]
            log_likelihood = exponent.sum(dim=1) - 0.5 * torch.log(2 * np.pi * self.variances[i]).sum()
            log_probabilities.append(prior + log_likelihood)
        
        log_probabilities = torch.stack(log_probabilities, dim=1)
        return torch.argmax(log_probabilities, dim=1)
    
    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return accuracy_score(y_test.numpy(), y_pred.numpy())

class NaiveBayesTrainer:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def train(self, X_train, y_train):
        model = NaiveBayesModel()
        model.fit(X_train, y_train, var_smoothing=self.var_smoothing)
        
        losses = []
        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train.numpy(), y_pred.numpy())
        losses.append(1 - accuracy)  # Store loss as (1 - accuracy)
        
        return model, losses

    def train_optuna(self, X_train, y_train, trial):
        var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1e-2, log=True)
        
        self.var_smoothing = var_smoothing
        model, _ = self.train(X_train, y_train)
        return model.score(X_train, y_train)
    
    def tune_hyperparameters(self, X_train, y_train, n_trials=10):
        sampler = TPESampler(seed=DEFAULT_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: self.train_optuna(X_train, y_train, trial), n_trials=n_trials)
        
        print("Best Hyperparameters:", study.best_params)
        return study.best_params
