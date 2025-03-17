import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import optuna
from optuna.samplers import TPESampler

from src.utils import set_seed, DEFAULT_SEED

set_seed()


class GeneticAlgorithmModel(nn.Module):
    def __init__(self, hidden_size=15):
        super(GeneticAlgorithmModel, self).__init__()
        self.fc1 = nn.Linear(26, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


class GeneticAlgorithmTrainer:
    def __init__(
        self,
        population_size=20,
        mutation_rate=0.1,
        generations=50,
        selection_rate=0.5,
        hidden_size=15,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.selection_rate = selection_rate
        self.hidden_size = hidden_size

    def initialize_individual(self, model):
        return [
            param.data.clone() + torch.randn_like(param) * 0.1
            for param in model.parameters()
        ]

    def evaluate_fitness(self, model, individual, X_train, y_train):
        with torch.no_grad():
            for param, ind_param in zip(model.parameters(), individual):
                param.data.copy_(ind_param)
            outputs = model(X_train).squeeze()
            predictions = (outputs > 0.5).int()
            accuracy = (predictions == y_train).float().mean().item()
        return float(accuracy)

    def select_parents(self, population, fitness_scores):
        fitness_scores = [float(score) for score in fitness_scores]
        sorted_population = [
            x
            for _, x in sorted(
                zip(fitness_scores, population), reverse=True, key=lambda pair: pair[0]
            )
        ]
        num_selected = max(1, int(len(population) * self.selection_rate))
        return sorted_population[:num_selected]

    def train(self, X_train, y_train):
        model = GeneticAlgorithmModel(hidden_size=self.hidden_size)
        population = [
            self.initialize_individual(model) for _ in range(self.population_size)
        ]
        best_fitness = 0.0
        losses = []

        for generation in range(self.generations):
            fitness_scores = [
                self.evaluate_fitness(model, ind, X_train, y_train)
                for ind in population
            ]
            gen_best_fitness = max(fitness_scores)
            best_fitness = max(best_fitness, gen_best_fitness)

            print(f"Generation {generation+1} - Best Fitness: {gen_best_fitness:.4f}")
            losses.append(
                1 - gen_best_fitness
            )  # Store loss as (1 - fitness) for tracking

            parents = self.select_parents(population, fitness_scores)
            next_generation = [
                self.initialize_individual(model) for _ in range(self.population_size)
            ]
            population = next_generation

        best_individual = self.select_parents(population, fitness_scores)[0]
        for param, best_param in zip(model.parameters(), best_individual):
            param.data.copy_(best_param)
        return model, losses

    def train_optuna(self, X_train, y_train, trial):
        population_size = trial.suggest_int("population_size", 10, 100)
        mutation_rate = trial.suggest_float("mutation_rate", 0.01, 0.5)
        generations = trial.suggest_int("generations", 10, 200)
        selection_rate = trial.suggest_float("selection_rate", 0.1, 0.9)
        hidden_size = trial.suggest_int("hidden_size", 1, 32)

        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.selection_rate = selection_rate
        self.hidden_size = hidden_size

        model, losses = self.train(X_train, y_train)

        with torch.no_grad():
            outputs = model(X_train).squeeze()
            predictions = (outputs > 0.5).int()
            accuracy = (predictions == y_train).float().mean().item()
        return -accuracy

    def tune_hyperparameters_ga(self, X_train, y_train, n_trials=10):
        sampler = TPESampler(seed=DEFAULT_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(
            lambda trial: self.train_optuna(X_train, y_train, trial), n_trials=n_trials
        )
        print("Best Hyperparameters:", study.best_params)
        return study.best_params
