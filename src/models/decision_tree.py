import os
import torch
import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, TensorDataset

# Importing modules from your project
from src.utils import set_seed, DEFAULT_SEED, set_log

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


search_space = {
    "criterion": ["gini", "entropy"],  # Criterion choices
    "max_depth": [2, 30],  # Max depth range
    "min_samples_split": [2, 20],  # Min samples split range
    "min_samples_leaf": [1, 10],  # Min samples leaf range
}


class DecisionTreeModel:
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

    def train(self, X_train, y_train):
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self.model.predict_proba(X)


class DecisionTreeTrainer:
    def __init__(
        self,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = DecisionTreeModel(
            criterion, max_depth, min_samples_split, min_samples_leaf
        )

    def train(self, X_train, y_train, X_val, y_val):
        # Train the model on the entire training set
        model = self.model.train(X_train, y_train)

        # Predict and evaluate on the training and validation sets
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        # Return the model and losses
        return model, train_loss, val_loss

    def train_optuna(self, X_train, y_train, X_val, y_val, trial):
        criterion = trial.suggest_categorical("criterion", search_space["criterion"])
        max_depth = trial.suggest_int(
            "max_depth", search_space["max_depth"][0], search_space["max_depth"][1]
        )
        min_samples_split = trial.suggest_int(
            "min_samples_split",
            search_space["min_samples_split"][0],
            search_space["min_samples_split"][1],
        )
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            search_space["min_samples_leaf"][0],
            search_space["min_samples_leaf"][1],
        )

        # Set model parameters
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        _, _, val_loss = self.train(X_train, y_train, X_val, y_val)
        return val_loss
