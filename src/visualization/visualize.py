import os
import sys
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    auc,
)
import numpy as np


# Importing modules from your project
from src.utils import (
    set_seed,
    DEFAULT_SEED,
    set_log,
    load_study,
    load_evaluation,
    load_training,
)

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


# Function to save a plot
def save_plot(plot_filename):
    plt.savefig(plot_filename)
    plt.close()

    logger.info(f"Plot saved as {plot_filename}")


# Function to visualize the Optuna study optimization history
def visualize_optimization_history(study, plot_dir):
    """
    Visualizes the optimization history for the Optuna study.

    :param study: The Optuna study object.
    :param plot_dir: Directory where the plot will be saved.
    """
    logger.info(f"Plotting optimization history for study.")
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    plot_filename = os.path.join(plot_dir, "st_optimization_history.png")
    save_plot(plot_filename)


# Function to visualize the parameter slice plot
def visualize_slice_plot(study, plot_dir):
    """
    Visualizes the slice plot for the Optuna study.

    :param study: The Optuna study object.
    :param plot_dir: Directory where the plot will be saved.
    """
    logger.info(f"Plotting slice plot for study.")
    fig = optuna.visualization.matplotlib.plot_slice(study)
    plot_filename = os.path.join(plot_dir, "st_slice_plot.png")
    save_plot(plot_filename)


# Function to visualize the parameter importance plot
def visualize_param_importance(study, plot_dir):
    """
    Visualizes the parameter importance for the Optuna study.

    :param study: The Optuna study object.
    :param plot_dir: Directory where the plot will be saved.
    """
    logger.info(f"Plotting parameter importance for study.")
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    plot_filename = os.path.join(plot_dir, "st_parameter_importance.png")
    save_plot(plot_filename)


# Main function to visualize the Optuna study results
def visualize_study(file_name):
    logger.info(f"Loading Optuna study from file: {file_name}")
    study = load_study(file_name)

    # Define the directory where plots will be saved
    plot_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "storage", "plots", file_name
    )
    os.makedirs(plot_dir, exist_ok=True)

    # Visualize each aspect of the study
    visualize_optimization_history(study, plot_dir)
    visualize_slice_plot(study, plot_dir)
    visualize_param_importance(study, plot_dir)


# Function to visualize the confusion matrix
def visualize_confusion_matrix(true_labels, predicted_labels, plot_dir):
    """
    Visualizes and saves the confusion matrix.

    :param true_labels: True labels of the test set.
    :param predicted_labels: Predicted labels from the model.
    :param plot_dir: Directory where the confusion matrix plot will be saved.
    """
    logger.info(f"Plotting confusion matrix.")
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Class 0", "Class 1"]
    )

    cm_plot_filename = os.path.join(plot_dir, "ev_confusion_matrix.png")
    cm_display.plot(cmap=plt.cm.Blues)
    save_plot(cm_plot_filename)


# Function to visualize the ROC curve
def visualize_roc_curve(true_labels, probabilities_labels, plot_dir):
    """
    Visualizes and saves the ROC curve and AUC score.

    :param true_labels: True labels of the test set.
    :param probabilities_labels: Predicted probabilities from the model.
    :param plot_dir: Directory where the ROC curve plot will be saved.
    """
    logger.info(f"Plotting ROC curve and AUC score.")
    # Compute the ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities_labels)
    roc_auc = auc(fpr, tpr)

    # Calculate Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr

    # Find the index of the best threshold
    best_threshold_index = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_index]

    # Get the TPR and FPR corresponding to the best threshold
    best_fpr = fpr[best_threshold_index]
    best_tpr = tpr[best_threshold_index]

    # Plot the ROC curve
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")  # Diagonal line

    plt.scatter(
        best_fpr,
        best_tpr,
        color="red",
        marker="o",
        s=100,
        label=f"Optimal Threshold (Threshold = {best_threshold:.2f})",
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")

    roc_curve_filename = os.path.join(plot_dir, "ev_roc_curve.png")
    save_plot(roc_curve_filename)


# Function to visualize the Precision-Recall curve
def visualize_precision_recall_curve(true_labels, probabilities_labels, plot_dir):
    """
    Visualizes and saves the Precision-Recall curve.

    :param true_labels: True labels of the test set.
    :param probabilities_labels: Predicted probabilities from the model.
    :param plot_dir: Directory where the Precision-Recall curve plot will be saved.
    """
    logger.info(f"Plotting Precision-Recall curve.")
    precision_vals, recall_vals, _ = precision_recall_curve(
        true_labels, probabilities_labels
    )

    auc_score = auc(recall_vals, precision_vals)

    plt.figure()
    plt.plot(
        recall_vals,
        precision_vals,
        label=f"Precision-Recall Curve (AUC = {auc_score:.2f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True)

    # Save the plot
    pr_curve_filename = os.path.join(plot_dir, "ev_precision_recall_curve.png")
    save_plot(pr_curve_filename)


# Main function to visualize evaluation results
def visualize_evaluate(file_name):
    logger.info(f"Loading evaluation results from file: {file_name}")

    # Load the saved evaluation results
    data = load_evaluation(file_name)
    predicted_labels = data["predictions"]
    true_labels = data["true_labels"]
    probabilities_labels = data["probabilities"]

    # Define the directory where the plot will be saved
    plot_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "storage", "plots", file_name
    )
    os.makedirs(plot_dir, exist_ok=True)

    # Visualize confusion matrix and ROC curve
    visualize_confusion_matrix(true_labels, predicted_labels, plot_dir)
    visualize_roc_curve(true_labels, probabilities_labels, plot_dir)
    visualize_precision_recall_curve(true_labels, probabilities_labels, plot_dir)


def visualize_loss_curve(train_losses, val_losses, plot_dir):
    logger.info(f"Plotting loss curves.")

    # Plot the training loss and validation loss on the same plot
    plt.plot(train_losses, label="Training Loss", color="blue")
    plt.plot(val_losses, label="Validation Loss", color="orange")

    # Labeling the plot
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")

    # Adding a legend
    plt.legend()

    # Save the plot
    loss_curve_filename = os.path.join(plot_dir, "tr_loss_curve.png")
    save_plot(loss_curve_filename)


def visualize_train(file_name):
    logger.info(f"Loading training results from file: {file_name}")
    # Load the saved evaluation results
    data = load_training(file_name)
    train_losses = data["train_losses"]
    val_losses = data["val_losses"]

    # Define the directory where the plot will be saved
    plot_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "storage", "plots", file_name
    )
    os.makedirs(plot_dir, exist_ok=True)

    visualize_loss_curve(train_losses, val_losses, plot_dir)


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
