import pandas as pd
import os

# Define the input data path
processed_data_path = os.path.join(os.path.dirname(__file__),"..", "..", "data", "processed", "processed_data.csv")
def get_features_and_labels():
    df = pd.read_csv(processed_data_path)
    X = df.drop('Churn', axis=1)  # Drop the label column, keep only the features
    y = df['Churn'] 
    return X, y 

