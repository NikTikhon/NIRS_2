import torch
import numpy as np
import math


def get_yi(model, record, device):
    with torch.no_grad():
        record = record.to(device)
        model.eval()
        return model(record)


class WrongOperationOption(Exception):
    pass


def get_y_hat(y: np.ndarray, operation: str):
    if operation == "max":
        return np.array(y).max(axis=0, initial=-math.inf)
    elif operation == "mean":
        return np.array(y).mean(axis=0)
    else:
        raise WrongOperationOption("The operation can be either mean or max")