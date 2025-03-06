import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Importing modules from your project
from models.train_model import (
    train_and_save,
    train_study_and_save,
    tune_and_save,
    evaluate_model,
)
from visualization.visualize import visualize_study, visualize_evaluate
from utils import set_seed, DEFAULT_SEED, set_log


# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()

# Define the path to the processed data
data_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "processed_data.csv"
)


def load_data(data_path):
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("Churn", axis=1), df["Churn"], test_size=0.2, random_state=DEFAULT_SEED
    )
    return (
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train ML models for Telco Churn Prediction."
    )
    subparsers = parser.add_subparsers(dest="command", help="Choose a command")

    # Train Command
    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "decision_tree",
            "neural_network",
            "naive_bayes",
            "genetic_algorithm",
            "graphical_model",
        ],
        required=True,
        help="Specify the model to train",
    )
    train_parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model"
    )

    # Option to choose between manual input and study-based loading of hyperparameters
    train_parser.add_argument(
        "--mode",
        type=str,
        choices=["manual", "study"],
        required=True,
        help="Choose how to load hyperparameters: 'manual' to type in values, 'study' to load from a study",
    )

    # Hyperparameters (for manual mode)
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--hidden_size", type=int, default=15, help="Hidden layer size"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )

    # Study File Name (for study mode)
    train_parser.add_argument(
        "--file_name", type=str, help="File name for the study data"
    )

    # Tune Command
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tuning")
    tune_parser.add_argument(
        "--model_type",
        type=str,
        choices=[
            "decision_tree",
            "neural_network",
            "naive_bayes",
            "genetic_algorithm",
            "graphical_model",
        ],
        required=True,
        help="Specify the model to tune",
    )
    tune_parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model"
    )
    tune_parser.add_argument(
        "--trials", type=int, default=20, help="Number of tuning trials"
    )

    tune_parser.add_argument(
        "--direction",
        type=str,
        choices=["minimize", "maximize"],
        required=True,
        help="Minimize for loss optimization, maximize for accuracy",
    )

    # Visualize Command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize model performance"
    )
    visualize_parser.add_argument(
        "--mode",
        type=str,
        choices=["tune", "evaluate"],
        required=True,
        help="Mode of visualization",
    )
    visualize_parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Name of the file containing model data",
    )

    # # Evaluate Command
    # eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    # eval_parser.add_argument(
    #     "--file_name", type=str, required=True, help="Name of the file of the model"
    # )

    args = parser.parse_args()

    # Load data only once
    data_path = "data/processed/processed_data.csv"
    X_train, X_test, y_train, y_test = load_data(data_path)

    if args.command == "train":
        logger.info(f"Training model: {args.model_name}")
        if args.mode == "manual":
            train_and_save(
                model_type=args.model_type,
                model_name=args.model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                hidden_size=args.hidden_size,
            )
        elif args.mode == "study":
            train_study_and_save(
                model_type=args.model_type,
                model_name=args.model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                file_name=args.file_name,
            )
    elif args.command == "tune":
        logger.info(f"Tuning hyperparameters for model: {args.model_name}")
        tune_and_save(
            model_type=args.model_type,
            model_name=args.model_name,
            X_train=X_train,
            y_train=y_train,
            n_trials=args.trials,
            direction=args.direction,
        )
    elif args.command == "visualize":
        if args.mode == "tune":
            visualize_study(file_name=args.file_name)
        elif args.mode == "evaluate":
            visualize_evaluate(file_name=args.file_name)
    # elif args.command == "evaluate":
    #     evaluate_model(
    #         file_name=args.file_name,
    #         X_test=X_test,
    #         y_test=y_test,
    #     )
    else:
        parser.print_help()


# CLI Example:
# python src/main.py train --model_type neural_network --model_name neural_network_1 --mode manual --lr 0.001 --batch_size 32 --epochs 20 --hidden_size 15
# python src/main.py train --model_type neural_network --model_name neural_network_2 --mode study --file_name ne_neural_network_2
# python src/main.py tune --model_type neural_network --model_name neural_network_2 --trials 20 --direction maximize
# python src/main.py visualize --mode tune --file_name ne_neural_network_2
# python src/main.py evaluate --file_name nn_neural_network_1
# python src/main.py predict --model neural_network --input_data data/sample.csv
if __name__ == "__main__":
    main()
