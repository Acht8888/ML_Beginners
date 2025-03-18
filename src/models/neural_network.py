import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import sys

<<<<<<< HEAD
# Importing modules from your project
from src.utils import set_seed, DEFAULT_SEED, set_log

# Set the random seed for reproducibility
set_seed()
=======

# Importing modules from your project
from src.utils import set_log

>>>>>>> main

# Configure logging for better visibility in production
logger = set_log()

<<<<<<< HEAD
search_space = {
    "lr": [1e-5, 1e-1],  # Log scale for learning rate
    "batch_size": [32, 128],  # Batch size range
    "hidden_size": [8, 256],  # Hidden layer size range
    "epochs": [10, 200],  # Number of epochs range
}


class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size=26, hidden_size=15):
        super(NeuralNetworkModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU to the first layer
        x = self.relu(self.fc2(x))  # Apply ReLU to the second layer
        x = self.sigmoid(self.fc3(x))  # Apply Sigmoid to the output layer
        return x
=======

class NeuralNetworkModel(nn.Module):
    def __init__(
        self, input_size=26, hidden_size=15, num_hidden_layers=1, dropout_rate=0.5
    ):
        super(NeuralNetworkModel, self).__init__()

        # List to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))

        for _ in range(num_hidden_layers):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Dropout(p=dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())

        # Register all layers as a sequential model
        self.enc_red = nn.Sequential(*layers)

    def forward(self, x):
        return self.enc_red(x)
>>>>>>> main


class NeuralNetworkTrainer:
    def __init__(
        self,
        input_size=26,
        hidden_size=15,
<<<<<<< HEAD
        lr=0.001,
=======
        num_hidden_layers=1,
        dropout_rate=0.5,
        lr=1e-3,
        weight_decay=1e-4,
>>>>>>> main
        batch_size=32,
        epochs=100,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
<<<<<<< HEAD
        self.lr = lr
=======
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.weight_decay = weight_decay
>>>>>>> main
        self.batch_size = batch_size
        self.epochs = epochs

    def create_data_loaders(self, X_train, y_train, X_val, y_val):
        # Create DataLoader for batching
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
<<<<<<< HEAD
            worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
=======
>>>>>>> main
            drop_last=True,
        )

        # Create DataLoader for validation
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
<<<<<<< HEAD
            worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
=======
>>>>>>> main
            drop_last=True,
        )

        return train_loader, val_loader

    def train(self, X_train, y_train, X_val, y_val):
<<<<<<< HEAD
        model = NeuralNetworkModel(self.input_size, self.hidden_size)
=======
        # Initialize the model
        model = NeuralNetworkModel(
            self.input_size, self.hidden_size, self.num_hidden_layers, self.dropout_rate
        )

        # Create data loaders
>>>>>>> main
        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val
        )

        # Loss function and optimizer
<<<<<<< HEAD
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        train_losses = []  # To store training loss values
        val_losses = []  # To store validation loss values

        # Training loop
        for epoch in range(self.epochs):
            model.train()
            epoch_train_loss = 0

=======
        criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Lists to store losses for plotting
        train_losses = []
        val_losses = []

        # Training loop
        for epoch in range(self.epochs):
            model.train()  # Set model to training mode
            epoch_train_loss = 0.0
            epoch_train_samples = 0  # To accumulate the number of training samples

            # Training phase
>>>>>>> main
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()
<<<<<<< HEAD
                epoch_train_loss += loss.item()

            # Validation phase
            model.eval()
            epoch_val_loss = 0
=======

                # Accumulate loss and number of samples
                epoch_train_loss += loss.item() * batch_X.size(
                    0
                )  # Multiply by batch size
                epoch_train_samples += batch_X.size(0)

            # Average training loss for the epoch
            avg_train_loss = epoch_train_loss / epoch_train_samples
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()  # Set model to evaluation mode
            epoch_val_loss = 0.0
            epoch_val_samples = 0  # To accumulate the number of validation samples

>>>>>>> main
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
<<<<<<< HEAD
                    epoch_val_loss += loss.item()

            # Print progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {epoch_train_loss / len(train_loader):.4f}, "
                    f"Validation Loss: {epoch_val_loss / len(val_loader):.4f}"
                )

            # Store loss for plotting
            train_losses.append(epoch_train_loss / len(train_loader))
            val_losses.append(epoch_val_loss / len(val_loader))

        return model, train_losses, val_losses

    def train_optuna(self, X_train, y_train, X_val, y_val, trial):
=======

                    # Accumulate validation loss and number of samples
                    epoch_val_loss += loss.item() * batch_X.size(
                        0
                    )  # Multiply by batch size
                    epoch_val_samples += batch_X.size(0)

            # Average validation loss for the epoch
            avg_val_loss = epoch_val_loss / epoch_val_samples
            val_losses.append(avg_val_loss)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{self.epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}"
                )

        return model, train_losses, val_losses

    def train_optuna(self, X_train, y_train, X_val, y_val, trial, search_space):
        # Hyperparameter optimization via Optuna
>>>>>>> main
        lr = trial.suggest_float("lr", search_space["lr"][0], search_space["lr"][1])
        batch_size = trial.suggest_int(
            "batch_size", search_space["batch_size"][0], search_space["batch_size"][1]
        )
        hidden_size = trial.suggest_int(
            "hidden_size",
            search_space["hidden_size"][0],
            search_space["hidden_size"][1],
        )
<<<<<<< HEAD
        epochs = trial.suggest_int(
            "epochs", search_space["epochs"][0], search_space["epochs"][1]
        )

        model = model = NeuralNetworkModel(self.input_size, self.hidden_size)
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.epochs = epochs
=======
        num_hidden_layers = trial.suggest_int(
            "num_hidden_layers",
            search_space["num_hidden_layers"][0],
            search_space["num_hidden_layers"][1],
        )
        dropout_rate = trial.suggest_float(
            "dropout_rate",
            search_space["dropout_rate"][0],
            search_space["dropout_rate"][1],
        )
        epochs = trial.suggest_int(
            "epochs", search_space["epochs"][0], search_space["epochs"][1]
        )
        weight_decay = trial.suggest_float(
            "weight_decay",
            search_space["weight_decay"][0],
            search_space["weight_decay"][1],
        )

        # Update self with trial values
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.weight_decay = weight_decay

        model = NeuralNetworkModel(
            self.input_size, self.hidden_size, self.num_hidden_layers, self.dropout_rate
        )
>>>>>>> main

        train_loader, val_loader = self.create_data_loaders(
            X_train, y_train, X_val, y_val
        )

        # Loss function and optimizer
        criterion = nn.BCELoss()
<<<<<<< HEAD
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
=======
        optimizer = optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
>>>>>>> main

        total_val_loss = 0

        # Training loop
        for epoch in range(self.epochs):
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
<<<<<<< HEAD
=======
            epoch_val_samples = 0

>>>>>>> main
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
<<<<<<< HEAD
                    epoch_val_loss += loss.item()

            # Add epoch loss to the total loss
            total_val_loss += epoch_val_loss

        return total_val_loss / self.epochs


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
=======
                    epoch_val_loss += loss.item() * batch_X.size(0)
                    epoch_val_samples += batch_X.size(0)

            # Average validation loss for the epoch
            avg_val_loss = epoch_val_loss / epoch_val_samples
            total_val_loss += avg_val_loss

        return total_val_loss / self.epochs
>>>>>>> main
