import torch


from utils import set_seed


set_seed()


def predict(model, X):
    """
    Makes predictions on new data.

    :param X: Input features (NumPy array or Pandas DataFrame)
    :return: Predicted labels (0 or 1) as a NumPy array
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        outputs = model(X).squeeze()  # Get model outputs
        predicted = (outputs > 0.5).int()  # Convert probabilities to binary (0 or 1)

    return predicted
