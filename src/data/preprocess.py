import os
import pandas as pd

from src.utils import set_seed, set_log


# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def load_data(file_name_raw):
    """Load CSV file into a pandas DataFrame."""
    base_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(
        base_dir, "..", "..", "data", "raw", f"{file_name_raw}.csv"
    )

    try:
        df = pd.read_csv(raw_data_path)
        logger.info("Data successfully loaded.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def save_data(df, file_name_processed):
    """Save the processed DataFrame to a CSV file."""
    base_dir = os.path.dirname(__file__)
    processed_data_path = os.path.join(
        base_dir, "..", "..", "data", "processed", f"{file_name_processed}.csv"
    )

    try:
        df.to_csv(processed_data_path, index=False)
        logger.info(f"Processed data saved to {processed_data_path}.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise
