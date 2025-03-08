import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# Importing modules from your project
from data.preprocess import preprocess
from models.train_model import (
    train_and_save,
    train_study_and_save,
    tune_and_save,
)
from models.predict_model import evaluate_model, evaluate_model_opt_threshold
from visualization.visualize import visualize_study, visualize_evaluate, visualize_train
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

    # Preprocess Data Command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess new data")
    preprocess_parser.add_argument(
        "--file_name_raw",
        type=str,
        required=True,
        help="Name of the file containing raw data",
    )
    preprocess_parser.add_argument(
        "--file_name_processed",
        type=str,
        required=True,
        help="Name of the file containing processed data",
    )

    load_parser = subparsers.add_parser("load", help="Load data")
    load_parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Name of the file containing data",
    )

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

    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--mode",
        type=str,
        choices=["opt", "manual"],
        required=True,
        help="Mode of evaluation",
    )
    eval_parser.add_argument(
        "--threshold", type=float, help="Set a custom threshold for classification"
    )
    eval_parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model file"
    )

    # Visualize Command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize model performance"
    )
    visualize_parser.add_argument(
        "--mode",
        type=str,
        choices=["study", "evaluate", "train"],
        required=True,
        help="Mode of visualization",
    )
    visualize_parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="Name of the file containing data",
    )

    args = parser.parse_args()

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    while True:
        # Prompt user for input
        user_input = input("\nEnter command or type 'exit' to quit: ").strip()
        if user_input.lower() == "exit":
            logger.info("Exiting program...")
            break

        # Parse the input command
        try:
            args = parser.parse_args(user_input.split())

            if args.command == "preprocess":
                preprocess(args.file_name_raw, args.file_name_processed)
            elif args.command == "load":
                data_path = f"data/processed/{args.file_name}.csv"
                X_train, X_test, y_train, y_test = load_data(data_path)
            elif args.command == "train":
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
            elif args.command == "evaluate":
                logger.info(f"Evaluating for model: {args.model_name}")
                if args.mode == "opt":
                    evaluate_model_opt_threshold(
                        file_name=args.model_name,
                        X_test=X_test,
                        y_test=y_test,
                    )
                elif args.mode == "manual":
                    evaluate_model(
                        file_name=args.model_name,
                        X_test=X_test,
                        y_test=y_test,
                        threshold=args.threshold,
                    )
            elif args.command == "visualize":
                if args.mode == "study":
                    visualize_study(file_name=args.file_name)
                elif args.mode == "evaluate":
                    visualize_evaluate(file_name=args.file_name)
                elif args.mode == "train":
                    visualize_train(file_name=args.file_name)
            else:
                parser.print_help()
        except Exception as e:
            logger.error(f"Error processing command: {e}")


# Additional Example:
# python src/main.py train --model_type neural_network --model_name neural_network_manual --mode manual --lr 0.001 --batch_size 32 --epochs 20 --hidden_size 15

# CLI Example:
# Run the script
# python src/main.py

# Preprocess and load the data
# preprocess --file_name_raw raw_data --file_name_processed processed_data
# load --file_name processed_data

# Tune and visualize the study
# tune --model_type neural_network --model_name neural_network_study --trials 20 --direction minimize
# visualize --mode study --file_name ne_neural_network_study

# Use the optimal hyperparameters to create and train the model
# train --model_type neural_network --model_name neural_network_study --mode study --file_name ne_neural_network_study
# visualize --mode train --file_name ne_neural_network_study

# Find and use the optimal threshold to evaluate the model
# evaluate --mode opt --model_name ne_neural_network_study
# visualize --mode evaluate --file_name ne_neural_network_study

# Use the desired threshold to evaluate the model
# evaluate --mode manual --threshold 0.5 --model_name ne_neural_network_study
# visualize --mode evaluate --file_name ne_neural_network_study

# python src/main.py predict --model neural_network --input_data data/sample.csv
if __name__ == "__main__":
    main()
