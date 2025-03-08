import torch
import os
import sys
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import joblib


# Importing modules from your project
from utils import set_seed, DEFAULT_SEED, set_log, save_model, load_model, save_study

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def load_model_and_predict(model, X):
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        outputs = model(X).squeeze()
        predictions = (outputs > 0.5).int()  # Binary classification with 0.5 threshold

    return predictions, outputs


def calculate_metrics(y_true, y_pred, y_probs):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def save_evaluation_results(file_name, save_path, y_test, y_probs, y_pred):
    os.makedirs(save_path, exist_ok=True)
    save_file = os.path.join(save_path, "evaluation.pkl")

    joblib.dump(
        {"true_labels": y_test, "probabilities": y_probs, "predictions": y_pred},
        save_file,
    )
    logger.info(f"Evaluation results saved to {save_file}.")


def find_optimal_threshold(y_test, y_probs):
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    j_scores = tpr - fpr  # Youden's J statistic

    best_threshold_index = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_index]

    logger.info(f"Optimal Threshold (Maximizing Youden's J): {best_threshold:.4f}")

    return best_threshold


def evaluate_model(file_name, X_test, y_test, threshold):
    """
    Evaluate the model on the test set and log metrics.

    :param file_name: The name of the saved model
    :param X_test: Test features
    :param y_test: Test labels
    :param threshold: The decision threshold for classification (float between 0 and 1).
    :return: Accuracy score
    """
    # Load the model
    model = load_model(file_name)

    # Get predictions
    y_pred, y_probs = load_model_and_predict(model, X_test)

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_probs)

    # Log metrics
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {value:.4f}")

    # Define the directory where plots will be saved
    save_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "storage", "evaluations", file_name
    )

    # Save evaluation results
    save_evaluation_results(file_name, save_path, y_test, y_probs, y_pred)


def predict_model(model, X, threshold):
    """
    Evaluate the trained model on new data and return predicted labels.

    :param model: The trained model to be used for predictions.
    :param X: Input features (NumPy array or Pandas DataFrame).
    :return: Predicted labels as a NumPy array, where each element is 0 or 1.
    """
    model.eval()

    with torch.no_grad():
        outputs = model(X).squeeze()
        predicted = (outputs > threshold).int()

    return predicted


def evaluate_model_opt_threshold(file_name, X_test, y_test):
    """
    Evaluate the model on the test set using the optimal threshold.

    :param file_name: The name of the saved model to be loaded
    :param X_test: Test features (NumPy array or Pandas DataFrame)
    :param y_test: True labels for the test set
    :return: None
    """
    # Load the model
    model = load_model(file_name)

    # Make predictions
    _, y_probs = load_model_and_predict(model, X_test)

    # Find the optimal threshold
    best_threshold = find_optimal_threshold(y_test, y_probs)

    # Evaluate the model with the optimal threshold
    y_pred = (y_probs > best_threshold).int()

    # Log and save metrics
    metrics = calculate_metrics(y_test, y_pred, y_probs)
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {value:.4f}")

    # Save the evaluation results
    save_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "storage", "evaluations", file_name
    )
    save_evaluation_results(file_name, save_path, y_test, y_probs, y_pred)


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
