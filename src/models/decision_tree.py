
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
import optuna
from optuna.samplers import TPESampler
from utils import set_seed, DEFAULT_SEED, set_log
import torch

set_seed()
logger = set_log()

class DecisionTreeModel:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.model = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

    def train(self, X_train, y_train):
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.detach().cpu().numpy()
        if isinstance(y_train, torch.Tensor):
            y_train = y_train.detach().cpu().numpy()

        self.model.fit(X_train, y_train)
        return self.model

    def predict(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self.model.predict(X)

    def predict_proba(self, X):
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        return self.model.predict_proba(X)


class DecisionTreeTrainer:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model = DecisionTreeModel(criterion, max_depth, min_samples_split, min_samples_leaf, random_state)

    def train(self, X_train, y_train):
        model = self.model.train(X_train, y_train)
        y_pred = model.predict(X_train)
        acc = accuracy_score(y_train, y_pred)
        losses = log_loss(y_train, model.predict_proba(X_train))
        return model, losses

    def train_optuna(self, X_train, y_train, trial):
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 2, 30)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        model, losses = self.train(X_train, y_train)
        return losses

    def tune_hyperparameters(self, X_train, y_train, n_trials=10):
        sampler = TPESampler(seed=DEFAULT_SEED)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(lambda trial: self.train_optuna(X_train, y_train, trial), n_trials=n_trials)
        
        print("Best Hyperparameters:", study.best_params)
        best_params = study.best_params
