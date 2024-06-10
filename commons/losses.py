import numpy as np

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