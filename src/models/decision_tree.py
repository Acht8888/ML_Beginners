import numpy as np
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import log_loss
import os
from src.utils import set_seed, set_log

# Đặt seed để đảm bảo tính tái lập
set_seed()
logger = set_log()

# Tham số GridSearchCV
PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [5, 10, 15, 20, 50],
    "min_samples_split": [2, 5, 10, 20, 50],
    "min_samples_leaf": [2, 5, 10, 20, 50],
    "class_weight": ["balanced", None],
}

class DecisionTreeModel:
    def __init__(self, criterion="gini", max_depth=10, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

class DecisionTreeTrainer:
    def __init__(
        self,
        criterion="gini",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
    ):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.best_params = None
        self.best_alpha = None

    def train(self, X_train, y_train, X_val, y_val):
        model = DecisionTreeModel(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )

        model.train(X_train, y_train)

        train_loss = log_loss(y_train, model.predict_proba(X_train))
        val_loss = log_loss(y_val, model.predict_proba(X_val))

        # Trả về model thay vì self.model
        return model, train_loss, val_loss

    def train_grid_search(self, X_train, y_train):
        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=42), PARAM_GRID, cv=5, scoring="accuracy"
        )
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        logger.info(f"Best Parameters: {self.best_params}")
        return self.best_params

    def post_pruning(self, X_train, y_train):
        if not self.best_params:
            raise ValueError("Cần chạy GridSearch trước khi post-pruning!")

        model = DecisionTreeClassifier(**self.best_params, random_state=42)
        path = model.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas[:-1]
        best_score = -np.inf

        for alpha in ccp_alphas:
            prune_model = DecisionTreeClassifier(**self.best_params, random_state=42, ccp_alpha=alpha)
            score = np.mean(cross_val_score(prune_model, X_train, y_train, cv=5, scoring="accuracy"))

            if score > best_score:
                best_score = score
                self.best_alpha = alpha
                best_model = prune_model

        best_model.fit(X_train, y_train)
        logger.info(f"Best ccp_alpha: {self.best_alpha}")
        return best_model