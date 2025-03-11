import os
import sys
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler

# Importing modules from your project
from models.neural_network import (
    train_nn,
    train_nn_optuna,
)

from models.decision_tree import (
    DecisionTreeTrainer,
)

from models.genetic_algorithm import GeneticAlgorithmTrainer

from utils import (
    set_seed,
    DEFAULT_SEED,
    set_log,
    save_model,
    save_study,
    load_study,
    save_training,
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
        trainer= DecisionTreeTrainer(
            max_depth=kwargs.get("max_depth", None),
            min_samples_split=kwargs.get("min_samples_split", 2),
            min_samples_leaf=kwargs.get("min_samples_leaf", 1),
            criterion=kwargs.get("criterion", "gini"),
            random_state=kwargs.get("random_state", 42),
        )
        print(f"Training {model_name} using Decision Tree...")
        model, losses = trainer.train(X_train, y_train)
    elif model_type == "neural_network":
        model, losses = train_nn(X_train=X_train, y_train=y_train, **kwargs)
    elif model_type == "naive_bayes":
        pass
    elif model_type == "genetic_algorithm":
        trainer = GeneticAlgorithmTrainer(
            population_size=kwargs.get("population_size", 50),
            mutation_rate=kwargs.get("mutation_rate", 0.05),
            generations=kwargs.get("generations", 100),
            selection_rate=kwargs.get("selection_rate", 0.2),
            hidden_size=kwargs.get("hidden_size", 15),
        )
        print(f"Training {model_name} using Genetic Algorithm...")
        model, losses = trainer.train(X_train, y_train)
    elif model_type == "graphical_model":
        pass
    else:
        logger.error(f"Model '{model_type}' not recognized!")
        raise ValueError(f"Model '{model_type}' not recognized!")

    if model:
        save_training(losses, model_type, model_name)
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
        trainer = DecisionTreeTrainer()
        best_params = trainer.tune_hyperparameters(X_train, y_train, n_trials=n_trials)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=DEFAULT_SEED))
        study.set_user_attr("best_params", best_params)
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
        trainer = GeneticAlgorithmTrainer()
        best_params = trainer.tune_hyperparameters_ga(X_train, y_train, n_trials=n_trials)
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=DEFAULT_SEED))
        study.set_user_attr("best_params", best_params)
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
