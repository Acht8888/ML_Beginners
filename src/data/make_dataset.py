import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV

# Importing modules from your project
from src.utils import set_seed, DEFAULT_SEED, set_log


# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def create_and_split_dataset(file_name):
    logger.info(f"Loading data: {file_name}")
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        base_dir, "..", "..", "data", "processed", f"{file_name}.csv"
    )
    df = pd.read_csv(data_path)

    # Split data into training (60%) and temporary (40%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        df.drop("Churn", axis=1), df["Churn"], test_size=0.4, random_state=DEFAULT_SEED
    )

    # Split the temporary data into validation (20%) and test (20%) sets
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=DEFAULT_SEED
    )

    # Convert to PyTorch tensors
    return (
        torch.tensor(X_train.values, dtype=torch.float32),
        torch.tensor(X_val.values, dtype=torch.float32),
        torch.tensor(X_test.values, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
        torch.tensor(y_test.values, dtype=torch.float32),
    )


def create_dataset(file_name):
    logger.info(f"Loading data: {file_name}")
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(
        base_dir, "..", "..", "data", "processed", f"{file_name}.csv"
    )
    df = pd.read_csv(data_path)

    # Extract features (X) and labels (y)
    X = df.drop("Churn", axis=1)  # Features (all columns except 'Churn')
    y = df["Churn"]  # Labels (Churn column)

    # Return tensors for X and y
    return (
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32),
    )
    
def print_results_table(results_list: list[dict[str, np.ndarray]], headings: list[str]) -> None:
    """
    Print table of sklearn cross-validation scores.
    Makes it easy to compare different models.

    :param results_list: cross-validation outputs of each model
    :param headings: heading that should be displayed for each model
    """
    left_col_size = 25
    other_col_size = 16
    # (name to print, name in dict)
    properties_to_print = [
        ("avg. accuracy:", "test_accuracy"),
        ("avg. f1_macro:", "test_f1_macro"),
        ("avg. precision_macro:", "test_precision_macro"),
        ("avg. recall_macro:", "test_recall_macro"),
        ("avg. fitting time:", "fit_time"),
        ("avg. score time:", "score_time")
    ]

    print(f"{'': <{left_col_size}}", end=" ")
    for heading in headings:
        print(f"{heading: <{other_col_size}}", end=" ")
    print("")

    for prop in properties_to_print:
        print(f"{prop[0]: <{left_col_size}}", end=" ")
        for results in results_list:
            print(f"{results[prop[1]].mean():2.4f}{'': <{other_col_size - 6}}", end=" ")
        print("")


def plot_gridsearch_scores(grid: GridSearchCV,
                           scoring_names: list[str],
                           plot_scores: bool = True,
                           plot_fit_time: bool = True,
                           plot_score_time: bool = True,
                           plot_stds: bool = False,
                           figsize: tuple[float, float] = (20, 5),
                           strip_param_prefix: str = ""
                           ) -> None:
    """
    Plot the hyperparameters of gridsearch.
    Inspired by: https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

    :param grid: GridSearchCV object from Gridsearch
    :param scoring_names: List of scores that should be plotted e.g. ["accuracy","f1_macro","precision_macro","recall_macro"]
    :param plot_scores: if scores should be plotted
    :param plot_fit_time: if fit time should be plotted
    :param plot_score_time: if score time should be plotted
    :param plot_stds: if the standard deviations should be shown in the graph
    :param figsize: figure size of the plots, this is the same as matplotlib's figsize
    :param strip_param_prefix: prefix to remove from hyperparameter names e.g turin M__LAPLACE_SMOOTHING to LAPLACE_SMOOTHING. This can be usefull since e.g. sklearn pipelines add prefixes, that will be shown in the plots.
    """

    def remove_prefix(text: str, prefix: str) -> str:
        if text.startswith(prefix):
            return text[len(prefix):]
        return text

    #

    # results from grid search
    results = grid.cv_results_
    mean_fit_time = results["mean_fit_time"]
    std_fit_time = results["std_fit_time"]
    mean_score_time = results["mean_score_time"]
    std_score_time = results["std_score_time"]
    scoring_results = []
    std_score_results = []
    for name in scoring_names:
        scoring_results.append(results['mean_test_' + name])
        std_score_results.append(results['std_test_' + name])

    # setup mask of the best scoring
    masks_names = list(grid.best_params_.keys())
    masks = []  # Save as mask, where each param has best value
    for param_name, best_value in grid.best_params_.items():
        masks.append(results["param_" + param_name].data == best_value)
    params = grid.param_grid

    # plotting Scores
    if plot_scores:
        if len(masks_names) == 1:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle('Score per Parameter')
            fig.text(0.04, 0.5, 'Mean Score', va='center', rotation='vertical')
            p = masks_names[0]
            x = [str(item) for item in params[p]]
            for j, scoring_name in enumerate(scoring_names):
                y = scoring_results[j]
                e = std_score_results[j]
                if plot_stds:
                    ax.errorbar(x, y, yerr=e, linestyle='--', marker='o', label=scoring_name, barsabove=True, capsize=5)
                else:
                    ax.errorbar(x, y, linestyle='--', marker='o', label=scoring_name)
            ax.set_xlabel(remove_prefix(p, strip_param_prefix).upper())
            plt.legend()
            plt.show()
        if len(masks_names) > 1:
            fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=figsize)
            fig.suptitle('Score per Parameter')
            fig.text(0.04, 0.5, 'Mean Score', va='center', rotation='vertical')
            for i, p in enumerate(masks_names):
                # find the best indices, for the given parameter
                # e.g. if param-1 has values 1,2,3; find the indices where they reach the best values
                m = np.stack(masks[:i] + masks[i + 1:])
                best_parms_mask = m.all(axis=0)
                best_indices = np.where(best_parms_mask)
                # plot
                # convert x to string, so that values are equally spaced in plot
                x = [str(item) for item in params[p]]
                for j, scoring_name in enumerate(scoring_names):
                    y = scoring_results[j][best_indices]
                    e = std_score_results[j][best_indices]
                    if plot_stds:
                        ax[i].errorbar(x, y, yerr=e, linestyle='--', marker='o', label=scoring_name, barsabove=True,
                                       capsize=5)
                    else:
                        ax[i].errorbar(x, y, linestyle='--', marker='o', label=scoring_name)
                ax[i].set_xlabel(remove_prefix(p, strip_param_prefix).upper())
            plt.legend()
            plt.show()

    # plotting fitting time
    if plot_fit_time:
        if len(masks_names) == 1:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle('Fitting Time per Parameter')
            fig.text(0.04, 0.5, 'Mean Fitting Time', va='center', rotation='vertical')
            p = masks_names[0]
            x = [str(item) for item in params[p]]
            y = mean_fit_time
            e = std_fit_time
            if plot_stds:
                ax.errorbar(x, y, yerr=e, linestyle='--', marker='o', color="#9467bd", barsabove=True, capsize=5)
            else:
                ax.errorbar(x, y, linestyle='--', marker='o', color="#9467bd")
            ax.set_xlabel(remove_prefix(p, strip_param_prefix).upper())
            plt.show()
        if len(masks_names) > 1:
            fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=figsize)
            fig.suptitle('Fitting Time per Parameter')
            fig.text(0.04, 0.5, 'Mean Fitting Time', va='center', rotation='vertical')
            for i, p in enumerate(masks_names):
                # find best indices
                m = np.stack(masks[:i] + masks[i + 1:])
                best_parms_mask = m.all(axis=0)
                best_indices = np.where(best_parms_mask)
                # plot
                x = [str(item) for item in params[p]]
                y = mean_fit_time[best_indices]
                e = std_fit_time[best_indices]
                if plot_stds:
                    ax[i].errorbar(x, y, yerr=e, linestyle='--', marker='o', color="#9467bd", barsabove=True, capsize=5)
                else:
                    ax[i].errorbar(x, y, linestyle='--', marker='o', color="#9467bd")
                ax[i].set_xlabel(remove_prefix(p, strip_param_prefix).upper())
            plt.show()

    # plotting score time
    if plot_score_time:
        if len(masks_names) == 1:
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle('Score Time per Parameter')
            fig.text(0.04, 0.5, 'Mean Score Time', va='center', rotation='vertical')
            p = masks_names[0]
            x = [str(item) for item in params[p]]
            y = mean_score_time
            e = std_score_time
            if plot_stds:
                ax.errorbar(x, y, yerr=e, linestyle='--', marker='o', color="#9467bd", barsabove=True, capsize=5)
            else:
                ax.errorbar(x, y, linestyle='--', marker='o', color="#9467bd")
            ax.set_xlabel(remove_prefix(p, strip_param_prefix).upper())
            plt.show()
        if len(masks_names) > 1:
            fig, ax = plt.subplots(1, len(params), sharex='none', sharey='all', figsize=figsize)
            fig.suptitle('Score Time per Parameter')
            fig.text(0.04, 0.5, 'Mean Score Time', va='center', rotation='vertical')
            for i, p in enumerate(masks_names):
                # find best indices
                m = np.stack(masks[:i] + masks[i + 1:])
                best_parms_mask = m.all(axis=0)
                best_indices = np.where(best_parms_mask)
                # plot
                x = [str(item) for item in params[p]]
                y = mean_score_time[best_indices]
                e = std_score_time[best_indices]
                if plot_stds:
                    ax[i].errorbar(x, y, yerr=e, linestyle='--', marker='o', color="#9467bd", barsabove=True, capsize=5)
                else:
                    ax[i].errorbar(x, y, linestyle='--', marker='o', color="#9467bd")
                ax[i].set_xlabel(remove_prefix(p, strip_param_prefix).upper())
            plt.show()