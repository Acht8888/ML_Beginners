import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from train_model import train_and_save, tune_hyperparameters, evaluate_model
from predict_model import predict
from utils import set_seed, DEFAULT_SEED

set_seed()

# Define the path to the processed data
data_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "processed_data.csv"
)


# Load Data
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
        "--model_type", type=str, required=True, help="Specify the model to train"
    )
    train_parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model"
    )
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--hidden_size", type=int, default=15, help="Hidden layer size"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )

    # Tune Command
    tune_parser = subparsers.add_parser("tune", help="Hyperparameter tuning")
    tune_parser.add_argument(
        "--model_type", type=str, required=True, help="Specify the model to tune"
    )
    tune_parser.add_argument(
        "--model_name", type=str, required=True, help="Name of the model"
    )
    tune_parser.add_argument(
        "--trials", type=int, default=20, help="Number of tuning trials"
    )

    # Evaluate Command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--file_name", type=str, required=True, help="Name of the file of the model"
    )

    args = parser.parse_args()

    # Load data only once
    data_path = "data/processed/processed_data.csv"
    X_train, X_test, y_train, y_test = load_data(data_path)

    if args.command == "train":
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
    elif args.command == "tune":
        tune_hyperparameters(
            model_type=args.model_type,
            model_name=args.model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_trials=args.trials,
        )
    elif args.command == "evaluate":
        evaluate_model(
            file_name=args.file_name,
            X_test=X_test,
            y_test=y_test,
        )
    else:
        parser.print_help()


# python src/models/main.py train --model_type neural_network --model_name neural_network_1 --lr 0.001 --batch_size 64 --epochs 50 --hidden_size 20
# python src/models/main.py tune --model_type neural_network --model_name neural_network_2 --trials 10
# python src/models/main.py evaluate --file_name nn_neural_network_1
# python src/models/main.py predict --model neural_network --input_data data/sample.csv
if __name__ == "__main__":
    main()
