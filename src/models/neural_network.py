import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import sys


# Importing modules from your project
from utils import set_seed, DEFAULT_SEED, set_log

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()

search_space = {
    "lr": [1e-5, 1e-1],  # Log scale for learning rate
    "batch_size": [32, 128],  # Batch size range
    "hidden_size": [8, 256],  # Hidden layer size range
    "epochs": [10, 100],  # Number of epochs range
}


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

    losses = []

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

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

        # Store loss for plotting
        losses.append(loss.item())

    return model, losses


def train_nn_optuna(X_train, y_train, trial):
    """
    Optimize hyperparameters using Optuna.

    :param X_train: Training features (Tensor).
    :param y_train: Training labels (Tensor).
    :param trial: Optuna trial object for hyperparameter tuning.
    :return: Final loss value after training.
    """
    lr = trial.suggest_float("lr", search_space["lr"][0], search_space["lr"][1])
    batch_size = trial.suggest_int(
        "batch_size", search_space["batch_size"][0], search_space["batch_size"][1]
    )
    hidden_size = trial.suggest_int(
        "hidden_size", search_space["hidden_size"][0], search_space["hidden_size"][1]
    )
    epochs = trial.suggest_int(
        "epochs", search_space["epochs"][0], search_space["epochs"][1]
    )

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


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
