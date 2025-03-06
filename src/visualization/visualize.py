import os
import optuna
import matplotlib.pyplot as plt

# Importing modules from your project
from utils import set_seed, DEFAULT_SEED, set_log, load_model, load_study

# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def visualize_study(file_name):
    """
    Visualizes the Optuna study results and saves plots to a specified directory.

    This function generates the following plots:
    1. Optimization history
    2. Parameter slice plot
    3. Parameter importance plot

    :param file_name: The name of the Optuna study file.
    """
    logger.info(f"Loading Optuna study from file: {file_name}")
    study = load_study(file_name)

    # Define the directory where plots will be saved
    plot_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "storage", "plots", file_name
    )
    os.makedirs(plot_dir, exist_ok=True)

    # Plot optimization history
    logger.info(f"Plotting optimization history for study: {file_name}")
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plot_filename = os.path.join(plot_dir, f"{file_name}_opt_hist.png")
    plt.savefig(plot_filename)
    logger.info(f"Optimization history plot saved as {plot_filename}")

    # Plot slice plot
    logger.info(f"Plotting slice plot for study: {file_name}")
    optuna.visualization.matplotlib.plot_slice(study)
    plot_filename = os.path.join(plot_dir, f"{file_name}_contour.png")
    plt.savefig(plot_filename)
    logger.info(f"Slice plot saved as {plot_filename}")

    # Plot parameter importance
    logger.info(f"Plotting parameter importance for study: {file_name}")
    optuna.visualization.matplotlib.plot_param_importances(study)
    plot_filename = os.path.join(plot_dir, f"{file_name}_param.png")
    plt.savefig(plot_filename)
    logger.info(f"Parameter importance plot saved as {plot_filename}")


def visualize_evaluate(file_name):
    pass
