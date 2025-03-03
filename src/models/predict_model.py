import sys
import os
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.train_model import get_train_test_data

# Get data from train_model
X_train, X_test, y_train, y_test= get_train_test_data()

# Load train model
with open("decision_tree_model.pkl", "rb") as f:
    data = pickle.load(f)
final_model = data["model"]

# Predict model
y_pred = final_model.predict(X_test)

# Saved model
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(model_dir, exist_ok=True)  # Đảm bảo thư mục tồn tại
predict_model_path = os.path.join(model_dir, "predict_model.pkl")
with open(predict_model_path, "wb") as f:
    pickle.dump({"model": final_model}, f)
print("Predict model saved at {predict_model_path}")

# Evaluation predict model
print("Final Optimized Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nFinal Classification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
