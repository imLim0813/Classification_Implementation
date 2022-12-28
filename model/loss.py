import torch.nn.functional as F


def cross_entropy(y_pred, y_true):
    return F.cross_entropy(y_pred, y_true)