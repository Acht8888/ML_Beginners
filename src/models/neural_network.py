import os
import torch
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


def train_nn(
    X_train, y_train, X_val, y_val, lr=0.001, batch_size=32, epochs=100, hidden_size=15
):
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

    # Create DataLoader for validation
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
        drop_last=False,
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []  # To store training loss values
    val_losses = []  # To store validation loss values

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()

        # Print progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_train_loss / len(train_loader):.4f}, "
                f"Validation Loss: {epoch_val_loss / len(val_loader):.4f}"
            )

        # Store loss for plotting
        train_losses.append(epoch_train_loss / len(train_loader))
        val_losses.append(epoch_val_loss / len(val_loader))

    return model, train_losses, val_losses


def train_nn_optuna(X_train, y_train, X_val, y_val, trial):
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

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
        drop_last=True,
    )

    # Create DataLoader for validation
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
        drop_last=False,
    )

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    total_val_loss = 0

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

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()

        # Add epoch loss to the total loss
        total_val_loss += epoch_val_loss

    return total_val_loss / epochs


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
