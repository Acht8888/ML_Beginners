import os
import pandas as pd

# Importing modules from your project
from src.features.build_features import (
    build_features,
)
from src.utils import set_seed, set_log


# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def get_data_paths(file_name_raw, file_name_processed):
    """Get paths for raw and processed data."""
    base_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(
        base_dir, "..", "..", "data", "raw", f"{file_name_raw}.csv"
    )
    processed_data_path = os.path.join(
        base_dir, "..", "..", "data", "processed", f"{file_name_processed}.csv"
    )
    return raw_data_path, processed_data_path


def load_data(file_path):
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info("Data successfully loaded.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def save_data(df, file_path):
    """Save the processed DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to {file_path}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def preprocess(file_name_raw, file_name_processed):
    """Main function to execute the data preprocessing pipeline."""
    raw_data_path, processed_data_path = get_data_paths(
        file_name_raw, file_name_processed
    )
    df = load_data(raw_data_path)
    df_processed = build_features(df)
    save_data(df_processed, processed_data_path)
