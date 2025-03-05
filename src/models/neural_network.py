import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler


from utils import set_seed, DEFAULT_SEED


set_seed()


class NeuralNetworkModel(nn.Module):
    def __init__(self, hidden_size=15):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(26, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU to the first layer
        x = self.relu(self.fc2(x))  # Apply ReLU to the second layer
        x = self.sigmoid(self.fc3(x))  # Apply Sigmoid to the output layer
        return x


def train_nn(
    model, X_train, y_train, lr=0.001, batch_size=32, epochs=100, hidden_size=15
):
    """
    Train the neural network with specified hyperparameters.

    Args:
        model (nn.Module): Neural network model.
        X_train (Tensor): Training features.
        y_train (Tensor): Training labels.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of training epochs.
    """
    model = NeuralNetworkModel(hidden_size=hidden_size)

    # Create DataLoader for batching
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Print average loss for this epoch
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")


def train_optuna(X_train, y_train, trial):
    """
    Train the neural network while optimizing hyperparameters using Optuna.

    Args:
        X_train (Tensor): Training features.
        y_train (Tensor): Training labels.
        trial (optuna.Trial): Optuna trial object.

    Returns:
        float: Final training loss.
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 8, 128)
    epochs = trial.suggest_int("epochs", 8, 128)
    hidden_size = trial.suggest_int("hidden_size", 1, 32)

    model = NeuralNetworkModel(hidden_size=hidden_size)

    # Create DataLoader for batching
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X).squeeze()

            if outputs.dim() != batch_y.dim():
                outputs = outputs.squeeze()  # Make sure output is 1D
                batch_y = batch_y.squeeze()  # Ensure target is also 1D

            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    return loss.item()


def tune_hyperparameters_nn(X_train, y_train, n_trials=10, direction="minimize"):
    """
    Run Optuna hyperparameter optimization.

    Args:
        train_func (function): Function to train the model.
        X_train (Tensor): Training features.
        y_train (Tensor): Training labels.
        n_trials (int): Number of tuning trials.
        direction (str): "minimize" for loss optimization, "maximize" for accuracy.

    Returns:
        optuna.study.Study: The Optuna study object.
    """
    sampler = TPESampler(seed=DEFAULT_SEED)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
    )
    study.optimize(
        lambda trial: train_optuna(X_train, y_train, trial),
        n_trials=n_trials,
    )

    print("Best Hyperparameters:", study.best_params)

    best_params = study.best_params
    best_lr = best_params["lr"]
    best_batch_size = best_params["batch_size"]
    best_epochs = best_params["epochs"]
    best_hidden_size = best_params["hidden_size"]

    return study, best_lr, best_batch_size, best_epochs, best_hidden_size
