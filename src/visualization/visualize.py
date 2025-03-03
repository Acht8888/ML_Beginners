import sys
import os
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pickle
from sklearn.metrics import roc_curve, roc_auc_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.models.train_model import get_train_test_data
from src.features.features import get_features_and_labels

# Get data from train_model
X_train, X_test, y_train, y_test= get_train_test_data()
X, y = get_features_and_labels()

# Visualize decision tree
with open("decision_tree_model.pkl", "rb") as f:
    data = pickle.load(f)
final_model = data["model"]
plt.figure(figsize=(20, 10))
plot_tree(final_model, feature_names=X.columns, class_names=['No Churn', 'Churn'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

# ROC curve
y_prob = final_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
