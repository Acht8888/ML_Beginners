import torch
import os
import sys
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# Importing modules from your project
from src.utils import (
    set_seed,
    DEFAULT_SEED,
    set_log,
    load_model,
    load_processed,
    save_evaluation,
    save_prediction,
)

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


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
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        y_probs = model(X_test).squeeze()
        y_pred = (y_probs > threshold).int()

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_probs)

    # Log metrics
    for metric_name, value in metrics.items():
        logger.info(f"{metric_name.capitalize()}: {value:.4f}")

    # Save evaluation results
    save_evaluation(file_name, y_test, y_probs, y_pred)


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

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        y_probs = model(X_test).squeeze()

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
    save_evaluation(file_name, y_test, y_probs, y_pred)


# # NOTE: WIP
# def evaluate_model_cross_validation(model, X, y, k=5):
#     """
#     Perform K-Fold Cross-Validation on the model.

#     :param model: The model to evaluate
#     :param X: Features
#     :param y: Labels
#     :param k: Number of folds for cross-validation (default: 5)
#     :return: Mean accuracy across K folds
#     """
#     kf = KFold(n_splits=k, shuffle=True, random_state=DEFAULT_SEED)
#     accuracies = []

#     for train_idx, test_idx in kf.split(X):
#         X_train, X_val = X[train_idx], X[test_idx]
#         y_train, y_val = y[train_idx], y[test_idx]

#         model.train_model(X_train, y_train)  # Retrain for each fold

#         # Evaluate model
#         accuracy = evaluate_model(model, X_val, y_val)
#         accuracies.append(accuracy)

#     mean_accuracy = np.mean(accuracies)
#     return mean_accuracy


def predict_model(model_name, file_name, threshold):
    """
    Evaluate the trained model on new data and return predicted labels.

    :param model: The trained model to be used for predictions.
    :param X: Input features (NumPy array or Pandas DataFrame).
    :return: Predicted labels as a NumPy array, where each element is 0 or 1.
    """
    model = load_model(model_name)
    model.eval()

    processed_data = load_processed(file_name)
    processed_data = processed_data.drop(columns=["Churn"], errors="ignore")
    processed_tensor = torch.tensor(processed_data.values, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(processed_tensor).squeeze()
        predicted = (outputs > threshold).int().numpy()

    processed_data["Predicted Churn"] = predicted

    save_prediction(processed_data, file_name)


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
