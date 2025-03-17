import os
import sys
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler

# Importing modules from your project
from src.models.decision_tree import DecisionTreeTrainer
from src.models.neural_network import NeuralNetworkTrainer
from src.models.genetic_algorithm import GeneticAlgorithmTrainer

from src.utils import (
    DEFAULT_SEED,
    set_log,
    save_model,
    save_study,
    load_study,
    save_training,
)


# Configure logging for better visibility in production
logger = set_log()


def train_and_save(model_type, model_name, X_train, y_train, X_val, y_val, **kwargs):
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
    trainer = None
    model = None

    print(f"Training {model_type} with the following hyperparameters:")
    for key, value in kwargs.items():
        print(f"{key}: {value}")

    # Select the model type and initialize it
    if model_type == "decision_tree":
        trainer = DecisionTreeTrainer(
            criterion=kwargs.get("criterion", "gini"),
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
        )
        model, train_losses, val_losses = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )
    elif model_type == "neural_network":
        trainer = NeuralNetworkTrainer(
            input_size=26,
            hidden_size=kwargs.get("hidden_size", 15),
            num_hidden_layers=kwargs.get("num_hidden_layers", 1),
            dropout_rate=kwargs.get("dropout_rate", 0.5),
            lr=kwargs.get("lr", 1e-3),
            weight_decay=kwargs.get("weight_decay", 1e-4),
            batch_size=kwargs.get("batch_size", 32),
            epochs=kwargs.get("epochs", 100),
        )
        model, train_losses, val_losses = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
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
        save_training(train_losses, val_losses, model_type, model_name)
        save_model(model, model_type, model_name)


def train_study_and_save(
    model_type, model_name, X_train, y_train, X_val, y_val, file_name
):
    study = load_study(file_name)

    train_and_save(
        model_type, model_name, X_train, y_train, X_val, y_val, **study.best_params
    )


def tune_and_save(
    model_type,
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    n_trials,
    direction,
    search_space,
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
        trainer = DecisionTreeTrainer()
        study = tune_hyperparameters(
            model_train_fn=trainer.train_optuna,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            direction=direction,
            search_space=search_space,
        )
    elif model_type == "neural_network":
        trainer = NeuralNetworkTrainer()
        study = tune_hyperparameters(
            model_train_fn=trainer.train_optuna,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            n_trials=n_trials,
            direction=direction,
            search_space=search_space,
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
    model_train_fn, X_train, y_train, X_val, y_val, n_trials, direction, search_space
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
        lambda trial: model_train_fn(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            trial=trial,
            search_space=search_space,
        ),
        n_trials=n_trials,
    )

    logger.info("Best Hyperparameters: %s", study.best_params)

    return study


if __name__ == "__main__":
    src_path = os.path.join(os.path.dirname(__file__), "..")
    sys.path.append(src_path)
