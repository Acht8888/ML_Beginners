import pandas as pd
import os

# Define path
raw_data_path = os.path.join(os.path.dirname(__file__),"..",  "..", "data", "raw", "raw_data.csv")

# Load raw data
df = pd.read_csv(raw_data_path)

# Show basic info
df.info()
