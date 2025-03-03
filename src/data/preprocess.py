import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Test
# Define the path to the raw data
raw_data_path = os.path.join(os.path.dirname(__file__),"..", "..", "data", "raw", "raw_data.csv")

# Define the path to the processed data
processed_data_path = os.path.join(os.path.dirname(__file__),"..", "..", "data", "processed", "processed_data.csv")

# Read the CSV file
df = pd.read_csv(raw_data_path)

# Drop customerID column
df.drop("customerID", axis="columns", inplace=True)

# Convert TotalCharges to numeric, handle errors as NaN
pd.to_numeric(df.TotalCharges, errors="coerce").isnull()

# Remove rows with space in TotalCharges
df = df[df.TotalCharges != " "]
df.TotalCharges = pd.to_numeric(df.TotalCharges)

# Replace "No internet service" and "No phone service" with "No" in one call
df.replace("No internet service", "No", inplace=True)
df.replace("No phone service", "No", inplace=True)


# Replace "Yes" with 1 and "No" with 0 for all yes/no columns
yes_no_columns = ["Partner", "Dependents", "PhoneService", "MultipleLines", 
                  "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
                  "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]
df[yes_no_columns] = df[yes_no_columns].replace({"Yes": 1, "No": 0})
df["gender"] = df["gender"].replace({"Female": 1, "Male": 0})
df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True).astype(int)

# Convert boolean columns to integers (True -> 1, False -> 0)
bool_columns = df.select_dtypes(include="bool").columns
df[bool_columns] = df[bool_columns].astype(int)

# Scale numerical columns
scaler = MinMaxScaler()
cols_to_scale = ["tenure", "MonthlyCharges", "TotalCharges"]
df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

# Save processed data
df.to_csv(processed_data_path, index=False)

