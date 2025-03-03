import sys
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.features.features import get_features_and_labels

# Split the dataset
from sklearn.model_selection import train_test_split
def get_train_test_data(test_size=0.2, random_state=42):
    X, y = get_features_and_labels() # Get data from features file
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test= get_train_test_data()

# Tune hyperparameters with expanded search
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [5, 10, 50],
    'min_samples_split': [2, 5, 10, 50],
    'min_samples_leaf': [2, 5, 10, 50],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train model with best parameters
best_model = DecisionTreeClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Post-pruning: Fine-tune ccp_alpha
path = best_model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas[:-1]
models = []
cv_scores = []
for ccp_alpha in ccp_alphas:
    model = DecisionTreeClassifier(**best_params, random_state=42, ccp_alpha=ccp_alpha)
    model.fit(X_train, y_train)
    score = np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))
    models.append(model)
    cv_scores.append(score)

# Select best ccp_alpha
best_alpha = ccp_alphas[np.argmax(cv_scores)]
print(f"Best ccp_alpha: {best_alpha}")
# Train final model with best ccp_alpha
final_model = DecisionTreeClassifier(**best_params, random_state=42, ccp_alpha=best_alpha)
final_model.fit(X_train, y_train)

# Create the path to the models directory
model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

# Đảm bảo thư mục model tồn tại
os.makedirs(model_dir, exist_ok=True)

# Save train_model
model_path = os.path.join(model_dir, "train_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({"model": final_model}, f)

print(f"Model trained and saved at {model_path}")
