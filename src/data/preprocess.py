import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Test
# Define the path to the raw data
raw_data_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "raw_data.csv"
)

# Define the path to the processed data
processed_data_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "processed_data.csv"
)

# Read the CSV file
df = pd.read_csv(raw_data_path)

# Drop customerID column
df.drop("customerID", axis="columns", inplace=True)

# Convert TotalCharges to numeric, handle errors as NaN
pd.to_numeric(df.TotalCharges, errors="coerce").isnull()

# Remove rows with space in TotalCharges
df1 = df[df.TotalCharges != " "]
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

# Replace "No internet service" and "No phone service" with "No" in one call
df1.replace("No internet service", "No", inplace=True)
df1.replace("No phone service", "No", inplace=True)

# Replace "Yes" with 1 and "No" with 0 for all yes/no columns
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
df1[yes_no_columns] = df1[yes_no_columns].replace({"Yes": 1, "No": 0})

# Replace gender values with 1 for Female, 0 for Male
df1["gender"] = df1["gender"].replace({"Female": 1, "Male": 0})

# Perform one-hot encoding for categorical columns
df2 = pd.get_dummies(data=df1, columns=["InternetService", "Contract", "PaymentMethod"])

# Scale numerical columns
cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

# Save the processed data to CSV
df2.to_csv(processed_data_path, index=False)
