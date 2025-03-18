import argparse
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
from src.utils import set_seed, DEFAULT_SEED, set_log

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def create_dataset(file_name, return_numpy=False):
    """
    Tạo dataset từ file CSV và chuyển đổi sang kiểu dữ liệu phù hợp.

    Args:
        file_name (str): Tên file CSV (không bao gồm đuôi .csv)
        return_numpy (bool): Nếu True, trả về NumPy arrays (cho Decision Tree).
                             Nếu False, trả về PyTorch Tensors (cho Neural Network).

    Returns:
        Tuple: (X_train, X_val, X_test, y_train, y_val, y_test) theo định dạng yêu cầu.
    """
    logger.info(f"Loading data: {file_name}")
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "..", "..", "data", "processed", f"{file_name}.csv")
    df = pd.read_csv(data_path)

    # Chia dữ liệu thành Train (60%) và Temp (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df.drop("Churn", axis=1).values, df["Churn"].values, test_size=0.4, random_state=DEFAULT_SEED
    )

    # Chia tiếp Temp thành Validation (20%) và Test (20%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=DEFAULT_SEED
    )

    if return_numpy:
        # Trả về NumPy arrays cho Decision Tree
        return (
            X_train.astype(np.float32),
            X_val.astype(np.float32),
            X_test.astype(np.float32),
            y_train.astype(np.int64),
            y_val.astype(np.int64),
            y_test.astype(np.int64),
        )
    else:
        # Trả về PyTorch Tensors cho Neural Network
        return (
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        )


