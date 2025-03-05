import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_data_paths():
    """Get paths for raw and processed data."""
    base_dir = os.path.dirname(__file__)
    raw_data_path = os.path.join(base_dir, "..", "..", "data", "raw", "raw_data.csv")
    processed_data_path = os.path.join(
        base_dir, "..", "..", "data", "processed", "processed_data.csv"
    )
    return raw_data_path, processed_data_path


def load_data(file_path):
    """Load CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logging.info("Data successfully loaded.")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise


def preprocess_data(df):
    """Preprocess the Telco Customer Churn dataset."""
    # Drop customerID column
    df.drop(columns=["customerID"], inplace=True, errors="ignore")

    # Convert TotalCharges to numeric, handle errors
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(
        subset=["TotalCharges"], inplace=True
    )  # Remove rows where TotalCharges could not be converted

    # Standardize categorical values
    df.replace({"No internet service": "No", "No phone service": "No"}, inplace=True)

    # Convert Yes/No columns to binary (1/0)
    yes_no_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "Churn",
    ]
    df[yes_no_columns] = df[yes_no_columns].replace({"Yes": 1, "No": 0})

    # Encode gender
    df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"])

    # Convert boolean columns to integers
    bool_columns = df.select_dtypes(include=bool).columns
    df[bool_columns] = df[bool_columns].astype(int)

    # Scale numerical columns
    cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    logging.info("Data preprocessing completed successfully.")
    return df


def save_data(df, file_path):
    """Save the processed DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
        logging.info(f"Processed data saved to {file_path}.")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        raise


def main():
    """Main function to execute the data preprocessing pipeline."""
    raw_data_path, processed_data_path = get_data_paths()
    df = load_data(raw_data_path)
    df_processed = preprocess_data(df)
    save_data(df_processed, processed_data_path)


if __name__ == "__main__":
    main()
