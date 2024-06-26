import numpy as np
from typing import Callable

def zero_one_loss(y_truth:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
    """Computes zero-one loss of the provided predicted labels with respect to
    the provided ground truth labels, returning a vector of loss values.

    Args:
        y_truth (np.ndarray): ground truth labels.
        y_pred (np.ndarray): predicted labels.

    Returns:
        np.ndarray: losses of every pair of labels.
    """

    return np.not_equal(y_truth, y_pred).astype(int)

def samples_error(predictor:Callable[[np.ndarray], np.ndarray], loss:Callable[[np.ndarray, np.ndarray], np.ndarray], y_truth:np.ndarray, data:np.ndarray) -> np.number:
    """Computes the prediction error of the given data and their true labels,
    using the specified predictor and loss function.

    Args:
        predictor (Callable[[np.ndarray], np.ndarray]): predictor with which
        to estimate labels of the data.
        loss (Callable[[np.ndarray, np.ndarray], np.ndarray]): function with
        which to compute individual prediction errors.
        y_truth (np.ndarray): ground truth labels of the data.
        data (np.ndarray): collection of samples.

    Returns:
        np.number: error computed on the samples.
    """

    samples_number = y_truth.size

    y_pred = predictor(data)
    error = np.sum(loss(y_truth, y_pred)) / samples_number

    return error
