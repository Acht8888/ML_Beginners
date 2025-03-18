import os
import sys
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold

# Importing modules from your project
from src.models.decision_tree import DecisionTreeTrainer
from src.models.neural_network import NeuralNetworkTrainer
from src.models.genetic_algorithm import GeneticAlgorithmTrainer
from src.models.predict_model import (
    evaluate_model_2,
    evaluate_model_opt_threshold,
    predict_model,
)

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


def cross_validation(model_type, X, y, k_folds, threshold, **kwargs):
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=DEFAULT_SEED)
    metrics_list = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "roc_auc": [],
    }

    if model_type == "decision_tree":
        pass
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

    elif model_type == "naive_bayes":
        pass
    elif model_type == "genetic_algorithm":
        pass
    elif model_type == "graphical_model":
        pass
    else:
        logger.error(f"Model '{model_type}' not recognized!")
        raise ValueError(f"Model '{model_type}' not recognized!")

    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        logger.info(f"Evaluating fold {fold + 1}/{k_folds}")

        # Split the data into train and validation for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model, train_losses, val_losses = trainer.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        # Evaluate model
        metrics = evaluate_model_2(model, X_val, y_val, threshold)
        metrics_list["accuracy"].append(metrics["accuracy"])
        metrics_list["precision"].append(metrics["precision"])
        metrics_list["recall"].append(metrics["recall"])
        metrics_list["f1"].append(metrics["f1"])
        metrics_list["roc_auc"].append(metrics["roc_auc"])

        # Log metrics for this fold
        logger.info(
            f"Fold {fold + 1} - Accuracy: {metrics["accuracy"]:.4f}, Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1: {metrics["f1"]:.4f}, ROC AUC: {metrics["roc_auc"]:.4f}"
        )

    # Calculate average metrics across all folds
    avg_metrics = {
        metric: sum(values) / k_folds for metric, values in metrics_list.items()
    }

    # Log averaged metrics
    logger.info(
        f"Average Metrics - Accuracy: {avg_metrics['accuracy']:.4f}, Precision: {avg_metrics['precision']:.4f}, Recall: {avg_metrics['recall']:.4f}, F1: {avg_metrics['f1']:.4f}, ROC AUC: {avg_metrics['roc_auc']:.4f}"
    )
    return avg_metrics


# Testing
