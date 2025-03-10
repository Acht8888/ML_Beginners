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

class DecisionTreeModel(nn.modules):
    def __init__(self, hidden_size=15):
        super(DecisionTreeModel, self)._init__()
        self.fc1 = nn.Linear(26, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

def train_nn(X_train, y_train, lr=0.001, batch_size=32, epochs=100, hidden_size=15):
    model = DecisionTreeModel(hidden_size=hidden_size)

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
        drop_last=True,
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = []

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

    model = DecisionTreeModel(hidden_size=hidden_size)

    dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda _: set_seed(DEFAULT_SEED),
        drop_last=True,
    )

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