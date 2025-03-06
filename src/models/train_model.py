import os
from sklearn.metrics import accuracy_score
import optuna
import matplotlib.pyplot as plt
import sys

# Importing modules from your project
from models.neural_network import NeuralNetworkModel, train_nn, tune_hyperparameters_nn
from models.predict_model import predict_model, evaluate_model
from models.utils import save_model, load_model
from utils import set_seed, DEFAULT_SEED, set_log

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def train_and_save(model_type, model_name, X_train, y_train, X_test, y_test, **kwargs):
    """
    Train and save the model of the specified type.

    :param model_type: The type of model (e.g., 'neural_network')
    :param model_name: Name for saving the model
    :param X_train: Training features
    :param y_train: Training labels
    :param X_test: Testing features
    :param y_test: Testing labels
    :param kwargs: Additional hyperparameters for the model
    """
    model = None

    # Select the model type and initialize it
    if model_type == "decision_tree":
        pass
        # Decision Tree code here (not implemented in provided code)
    elif model_type == "neural_network":
        model = train_nn(
            X_train=X_train,
            y_train=y_train,
            lr=kwargs.get("lr", 0.001),
            batch_size=kwargs.get("batch_size", 32),
            epochs=kwargs.get("epochs", 100),
            hidden_size=kwargs.get("hidden_size", 15),
        )
    elif model_type == "naive_bayes":
        pass
    elif model_type == "genetic_algorithm":
        pass
    elif model_type == "graphical_model":
        pass
    else:
        logger.error(f"Model '{model_type}' not recognized!")
        raise ValueError(f"Model '{model_type}' not recognized!")

    if model:
        # Evaluate and save the model
        predictions = predict_model(model, X_test)
        test_accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        save_model(model, model_type, model_name)


def tune_and_save(
    model_type,
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    n_trials,
    direction="minimize",
):
    """
    Tune hyperparameters using Optuna and save the best model.

    :param model_type: The type of the model
    :param model_name: The name of the model
    :param X_train: Training features
    :param y_train: Training labels
    :param X_test: Testing features
    :param y_test: Testing labels
    :param n_trials: Number of Optuna trials for hyperparameter tuning
    :param direction: Direction for optimization ('minimize' or 'maximize')
    """
    if direction not in ["minimize", "maximize"]:
        logger.error(f"Direction '{direction}' not recognized!")
        raise ValueError(f"Direction '{direction}' not recognized!")

    # Select the model
    if model_type == "decision_tree":
        pass
    elif model_type == "neural_network":
        study, best_lr, best_batch_size, best_epochs, best_hidden_size = (
            tune_hyperparameters_nn(
                X_train=X_train, y_train=y_train, n_trials=n_trials, direction=direction
            )
        )
    elif model_type == "naive_bayes":
        pass
    elif model_type == "genetic_algorithm":
        pass
    elif model_type == "graphical_model":
        pass
    else:
        logger.error(f"Model '{model_type}' not recognized!")
        raise ValueError(f"Model '{model_type}' not recognized!")

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


def visualize_study(study_name):
    study = study_name

    optuna.visualization.matplotlib.plot_optimization_history(study)

    plot_filename = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "storage",
        "plots",
        "optimization_history.png",
    )
    plt.savefig(plot_filename)


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
