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

from neural_network import (
    NeuralNetworkModel,
    train_nn,
    tune_hyperparameters_nn,
)
from predict_model import predict
from utils import set_seed, DEFAULT_SEED

set_seed()


# Define the path to save the model
model_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "experiments"
)


def train_and_save(model_type, model_name, X_train, y_train, X_test, y_test, **kwargs):

    # Select the model
    if model_type == "decision_tree":
        print("Decision Tree")
    elif model_type == "neural_network":
        model = NeuralNetworkModel(hidden_size=kwargs.get("hidden_size", 15))

        # Train the model
        print(f"Training {model_name} model...")
        train_nn(
            model,
            X_train,
            y_train,
            lr=kwargs.get("lr", 0.001),
            batch_size=kwargs.get("batch_size", 32),
            epochs=kwargs.get("epochs", 100),
            hidden_size=kwargs.get("hidden_size", 15),
        )

    elif model_type == "naive_bayes":
        print("Naive Bayes")
    elif model_type == "genetic_algorithm":
        print("Genetic Algorithm")
    elif model_type == "graphical_model":
        print("Graphical Model")
    else:
        raise ValueError(f"Model '{model_type}' not recognized!")

    # Evaluate the model
    predictions = predict(model, X_test)
    test_accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save the model
    save_model(model=model, model_type=model_type, model_name=model_name)


def save_model(model, model_type, model_name):
    # Select the model
    if model_type == "decision_tree":
        file_name = "dt_" + f"{model_name}"
    elif model_type == "neural_network":
        file_name = "nn_" + f"{model_name}"
    elif model_type == "naive_bayes":
        file_name = "nb_" + f"{model_name}"
    elif model_type == "genetic_algorithm":
        file_name = "ga_" + f"{model_name}"
    elif model_type == "graphical_model":
        file_name = "gm_" + f"{model_name}"
    else:
        raise ValueError(f"Model '{model_type}' not recognized!")

    final_model_path = os.path.join(model_path, file_name + ".pth")
    torch.save(model, final_model_path)
    print(f"Model saved to {final_model_path}")


def load_model(file_name):
    final_model_path = os.path.join(model_path, file_name + ".pth")
    loaded_model = torch.load(final_model_path, weights_only=False)
    loaded_model.eval()

    print(f"Model {file_name} loaded")
    return loaded_model


def evaluate_model(file_name, X_test, y_test):
    """
    Evaluates the model on the test set and returns accuracy.

    :param X_test: Test features (NumPy array or Pandas DataFrame)
    :param y_test: Test labels (NumPy array or Pandas Series)
    :return: Accuracy score
    """
    model = load_model(file_name=file_name)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        outputs = model(X_test).squeeze()  # Forward pass
        predicted = (outputs > 0.5).int()  # Convert probabilities to binary (0 or 1)

    # Convert PyTorch tensors to NumPy arrays for Scikit-learn
    accuracy = accuracy_score(y_test, predicted)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def evaluate_model_cross_validation(model, X, y, k=5):
    """
    Evaluates the model using K-Fold Cross-Validation.

    :param X: Features (PyTorch Tensor)
    :param y: Labels (PyTorch Tensor)
    :param k: Number of folds (default: 5)
    :return: Mean accuracy score across K folds
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=DEFAULT_SEED)
    accuracies = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        model.train_model(X_train, y_train)  # Retrain for each fold

        # Evaluate model
        accuracy = evaluate_model(model, X_val, y_val)
        accuracies.append(accuracy)

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy


def tune_hyperparameters(
    model_type,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials,
    direction="minimize",
):
    # Select the model
    if model_type == "decision_tree":
        print("Decision Tree")
    elif model_type == "neural_network":
        study, best_lr, best_batch_size, best_epochs, best_hidden_size = (
            tune_hyperparameters_nn(
                X_train=X_train,
                y_train=y_train,
                n_trials=n_trials,
            )
        )
        train_and_save(
            model_type=model_type,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            lr=best_lr,
            batch_size=best_batch_size,
            epochs=best_epochs,
            hidden_size=best_hidden_size,
        )
    elif model_type == "naive_bayes":
        print("Naive Bayes")
    elif model_type == "genetic_algorithm":
        print("Genetic Algorithm")
    elif model_type == "graphical_model":
        print("Graphical Model")
    else:
        raise ValueError(f"Model '{model_type}' not recognized!")
