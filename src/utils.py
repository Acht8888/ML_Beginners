import os
import torch
import joblib
import random
import logging
import numpy as np


# Define the path to the model
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "experiments")

# Define the path to the study
study_path = os.path.join(os.path.dirname(__file__), "..", "storage", "studies")


DEFAULT_SEED = 42


def set_seed(seed=DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_log():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)


# Set the random seed for reproducibility
set_seed()

# Configure logging for better visibility in production
logger = set_log()


def save_model(model, model_type, model_name):
    """
    Save the trained.

    :param model: The trained model
    :param model_type: The type of the model (used for naming the file)
    :param model_name: The name of the model (used for saving)
    """
    model_filename = f"{model_type[0:2]}_{model_name}.pth"
    final_model_path = os.path.join(model_path, model_filename)
    torch.save(model, final_model_path)
    logger.info(f"Model saved to {final_model_path}")


def load_model(file_name):
    """
    Load a saved model from the disk.

    :param file_name: The name of the saved model file
    :return: The loaded model
    """
    final_model_path = os.path.join(model_path, f"{file_name}.pth")
    model = torch.load(final_model_path)
    model.eval()
    logger.info(f"Model {file_name} loaded successfully.")
    return model


def save_study(study, model_type, model_name):
    study_filename = f"{model_type[0:2]}_{model_name}.pkl"
    final_study_path = os.path.join(study_path, study_filename)
    joblib.dump(study, final_study_path)
    logger.info(f"Model saved to {final_study_path}")


def load_study(file_name):
    final_study_path = os.path.join(study_path, f"{file_name}.pkl")
    study = joblib.load(final_study_path)

    return study
