import os
import sys
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler

# Importing modules from your project
from models.neural_network import (
    NeuralNetworkModel,
    train_nn,
    train_nn_optuna,
)
from models.predict_model import predict_model, evaluate_model
from visualization.visualize import visualize_study
from utils import (
    set_seed,
    DEFAULT_SEED,
    set_log,
    save_model,
    load_model,
    save_study,
    load_study,
)

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

    print(f"Training {model_type} with the following hyperparameters:")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

    # Select the model type and initialize it
    if model_type == "decision_tree":
        pass
    elif model_type == "neural_network":
        model = train_nn(X_train=X_train, y_train=y_train, **kwargs)
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
    model_type, model_name, X_train, y_train, n_trials, direction="minimize"
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
    study = None
    if direction not in ["minimize", "maximize"]:
        logger.error(f"Direction '{direction}' not recognized!")
        raise ValueError(f"Direction '{direction}' not recognized!")

    # Select the model
    if model_type == "decision_tree":
        pass
    elif model_type == "neural_network":
        study = tune_hyperparameters(
            model_train_fn=train_nn_optuna,
            X_train=X_train,
            y_train=y_train,
            n_trials=n_trials,
            direction=direction,
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

    if study:
        save_study(study, model_type, model_name)


def tune_hyperparameters(
    model_train_fn, X_train, y_train, n_trials=10, direction="minimize"
):
    """
    Tune hyperparameters of the neural network using Optuna.

    :param X_train: Training features (Tensor).
    :param y_train: Training labels (Tensor).
    :param n_trials: Number of trials for Optuna optimization. Default is 10.
    :param direction: Optimization direction ('minimize' or 'maximize'). Default is 'minimize'.
    :return: Optuna study object and the best hyperparameters.
    """
    sampler = TPESampler(seed=DEFAULT_SEED)
    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
    )
    study.optimize(
        lambda trial: model_train_fn(X_train, y_train, trial),
        n_trials=n_trials,
    )

    logger.info("Best Hyperparameters: %s", study.best_params)

    return study


def train_study_and_save(
    model_type, model_name, X_train, y_train, X_test, y_test, file_name
):
    study = load_study(file_name)

    train_and_save(
        model_type, model_name, X_train, y_train, X_test, y_test, **study.best_params
    )


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
