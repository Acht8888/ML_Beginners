import torch
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np


# Importing modules from your project
from utils import set_seed, DEFAULT_SEED, set_log, save_model, load_model, save_study

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def predict_model(model, X):
    """
    Evaluate the trained model on new data and return predicted labels.

    :param model: The trained model to be used for predictions.
    :param X: Input features (NumPy array or Pandas DataFrame).
    :return: Predicted labels as a NumPy array, where each element is 0 or 1.
    """
    model.eval()

    with torch.no_grad():
        outputs = model(X).squeeze()
        predicted = (outputs > 0.5).int()

    return predicted


def evaluate_model(file_name, X_test, y_test):
    """
    Evaluate the model on the test set and return accuracy.

    :param file_name: The name of the saved model
    :param X_test: Test features
    :param y_test: Test labels
    :return: Accuracy score
    """
    model = load_model(file_name)

    with torch.no_grad():
        outputs = model(X_test).squeeze()
        predicted = (outputs > 0.5).int()

    accuracy = accuracy_score(y_test, predicted)
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


# NOTE: WIP
def evaluate_model_cross_validation(model, X, y, k=5):
    """
    Perform K-Fold Cross-Validation on the model.

    :param model: The model to evaluate
    :param X: Features
    :param y: Labels
    :param k: Number of folds for cross-validation (default: 5)
    :return: Mean accuracy across K folds
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


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
