import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
from optuna.samplers import TPESampler
import sys


# Importing modules from your project
from utils import set_seed, DEFAULT_SEED, set_log

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


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


def train_nn(X_train, y_train, lr=0.001, batch_size=32, epochs=100, hidden_size=15):
    """
    Train a neural network model on the training data.

    :param X_train: Training features (Tensor).
    :param y_train: Training labels (Tensor).
    :param lr: Learning rate for the optimizer. Default is 0.001.
    :param batch_size: Batch size for training. Default is 32.
    :param epochs: Number of training epochs. Default is 100.
    :param hidden_size: Number of neurons in the hidden layers. Default is 15.
    :return: Trained model.
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
        drop_last=True,
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
            logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    return model


def train_optuna(X_train, y_train, trial):
    """
    Optimize hyperparameters using Optuna.

    :param X_train: Training features (Tensor).
    :param y_train: Training labels (Tensor).
    :param trial: Optuna trial object for hyperparameter tuning.
    :return: Final loss value after training.
    """
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 1, 128)
    epochs = trial.suggest_int("epochs", 1, 128)
    hidden_size = trial.suggest_int("hidden_size", 1, 128)

    model = NeuralNetworkModel(hidden_size=hidden_size)

    # Create DataLoader for batching
    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
        drop_last=True,
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

            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    return loss.item()


def tune_hyperparameters_nn(X_train, y_train, n_trials=10, direction="minimize"):
    """
    Tune hyperparameters of the neural network using Optuna.

    :param X_train: Training features (Tensor).
    :param y_train: Training labels (Tensor).
    :param n_trials: Number of trials for Optuna optimization. Default is 10.
    :param direction: Optimization direction ('minimize' or 'maximize'). Default is 'minimize'.
    :return: Optuna study object and the best hyperparameters.
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

    logger.info("Best Hyperparameters: %s", study.best_params)

    best_params = study.best_params
    best_lr = best_params["lr"]
    best_batch_size = best_params["batch_size"]
    best_epochs = best_params["epochs"]
    best_hidden_size = best_params["hidden_size"]

    return study, best_lr, best_batch_size, best_epochs, best_hidden_size


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
