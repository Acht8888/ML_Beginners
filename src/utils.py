<<<<<<< HEAD

=======
>>>>>>> main
import os
import torch
import joblib
import random
import logging
import numpy as np
import pandas as pd


model_path = os.path.join(os.path.dirname(__file__), "..", "models", "experiments")

study_path = os.path.join(os.path.dirname(__file__), "..", "storage", "studies")

evaluation_path = os.path.join(
    os.path.dirname(__file__), "..", "storage", "evaluations"
)

training_path = os.path.join(os.path.dirname(__file__), "..", "storage", "trainings")

prediction_path = os.path.join(os.path.dirname(__file__), "..", "data", "predicted")

processed_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

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


<<<<<<< HEAD
# Set the random seed for reproducibility
set_seed()

=======
>>>>>>> main
# Configure logging for better visibility in production
logger = set_log()


def save_model(model, model_type, model_name):
<<<<<<< HEAD
    if model_type == "decision_tree":
        model_filename = f"{model_type[0:2]}_{model_name}.pkl"
        final_model_path = os.path.join(model_path, model_filename)
        joblib.dump(model, final_model_path)
        logger.info(f"Decision Tree Model saved to {model_path}")
    else:
        model_filename = f"{model_type[0:2]}_{model_name}.pth"
        final_model_path = os.path.join(model_path, model_filename)
        torch.save(model, final_model_path)
        logger.info(f"Model saved to {final_model_path}")


def load_model(file_name):
    if file_name.startswith("de_decision_tree_manual") or file_name.startswith("de_decision_tree_study"):
       final_model_path = os.path.join(model_path, f"{file_name}.pkl")
       model = joblib.load(final_model_path)
    else:
       final_model_path = os.path.join(model_path, f"{file_name}.pth")
       model = torch.load(final_model_path, weights_only=False)
       model.eval()
       logger.info(f"Model {file_name} loaded successfully.")
=======
    model_filename = f"{model_type[0:2]}_{model_name}.pth"
    final_model_path = os.path.join(model_path, model_filename)
    torch.save(model, final_model_path)
    logger.info(f"Model saved to {final_model_path}")


def load_model(file_name):
    final_model_path = os.path.join(model_path, f"{file_name}.pth")
    model = torch.load(final_model_path, weights_only=False)
    model.eval()
    logger.info(f"Model {file_name} loaded successfully.")
>>>>>>> main
    return model


def save_study(study, model_type, model_name):
    study_filename = f"{model_type[0:2]}_{model_name}.pkl"
    final_study_path = os.path.join(study_path, study_filename)
    joblib.dump(study, final_study_path)
    logger.info(f"Study saved to {final_study_path}")


def load_study(file_name):
    final_study_path = os.path.join(study_path, f"{file_name}.pkl")
    logger.info(f"Study {file_name} loaded successfully.")
    return joblib.load(final_study_path)


def save_evaluation(file_name, y_test, y_probs, y_pred):
    os.makedirs(evaluation_path, exist_ok=True)
    evaluation_filename = f"{file_name}.pkl"
    final_evaluation_path = os.path.join(evaluation_path, evaluation_filename)
    joblib.dump(
        {"true_labels": y_test, "probabilities": y_probs, "predictions": y_pred},
        final_evaluation_path,
    )
    logger.info(f"Evaluation results saved to {final_evaluation_path}.")


def load_evaluation(file_name):
    final_evaluation_path = os.path.join(evaluation_path, f"{file_name}.pkl")
    logger.info(f"Evaluation {file_name} loaded successfully.")
    return joblib.load(final_evaluation_path)


def save_training(train_losses, val_losses, model_type, model_name):
    os.makedirs(training_path, exist_ok=True)
    training_filename = f"{model_type[0:2]}_{model_name}.pkl"
    final_training_path = os.path.join(training_path, training_filename)
    joblib.dump(
        {"train_losses": train_losses, "val_losses": val_losses}, final_training_path
    )
    logger.info(f"Training results saved to {final_training_path}.")


def load_training(file_name):
    final_training_path = os.path.join(training_path, f"{file_name}.pkl")
    logger.info(f"Training {file_name} loaded successfully.")
    return joblib.load(final_training_path)


def save_prediction(df, file_name):
    os.makedirs(prediction_path, exist_ok=True)
    prediction_filename = f"{file_name}.csv"
    final_prediction_path = os.path.join(prediction_path, prediction_filename)
    df.to_csv(final_prediction_path, index=False)
    logger.info(f"Prediction saved to {final_prediction_path}.")


def load_processed(file_name):
    final_processed_path = os.path.join(processed_path, f"{file_name}.csv")
    logger.info(f"Processed file {file_name} loaded successfully.")
    return pd.read_csv(final_processed_path)
