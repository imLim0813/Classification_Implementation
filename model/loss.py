import torch.nn.functional as F


def cross_entropy(y_pred, y_true):
    """
    A function to calculate loss.
    """
    return F.cross_entropy(y_pred, y_true)