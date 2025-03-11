import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Importing modules from your project
from utils import set_seed, set_log


# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def build_features(df):
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

    logger.info("Data preprocessing completed successfully.")
    return df
