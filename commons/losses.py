import numpy as np
import math
from typing import Callable
from typing import Any

from predictors.treepredictors import TreePredictor

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

    samples_number = len(y_truth)

    y_pred = predictor(data)
    error = np.sum(loss(y_truth, y_pred)) / samples_number

    return error

def create_folds(data:np.ndarray, folds_number:int) -> np.ndarray[np.ndarray]:
    data_size = len(data)
    fold_size = math.ceil(data_size / folds_number)

    data_folds = [data[i*fold_size:(i+1)*fold_size] for i in range(folds_number)]
    data_folds = np.array(data_folds)

    return data_folds

def cross_validation(model:TreePredictor, loss:Callable[[np.ndarray, np.ndarray], np.ndarray], data_folds:np.ndarray[np.ndarray], labels_folds:np.ndarray[np.ndarray], verbose:bool = False) -> np.number:
    folds_number = len(data_folds)
    folds_losses = []

    if verbose: print(f'Running cross validation on {folds_number} folds of size {len(data_folds[0])}')

    ## Iterate on folds
    for k in range(folds_number):
        if verbose: print(f'Fold {k} iteration:')

        ## Split folds
        train_data = np.concatenate(data_folds[np.arange(folds_number) != k], axis=0)
        train_labels = np.concatenate(labels_folds[np.arange(folds_number) != k])
        test_data = data_folds[k]
        test_labels = labels_folds[k]
        fold_size = len(test_labels)

        ## Train model
        if verbose: print(f'Training model on folds [-{k}]...')

        model.fit(train_data, train_labels)

        ## Test model
        if verbose: print(f'Testing model on fold [{k}]...')

        predictions = model.predict(test_data)
        fold_loss = loss(test_labels, predictions)
        fold_loss = np.sum(fold_loss) / fold_size

        if verbose: print(f'> fold [{k}] error: {fold_loss}')

        folds_losses.append(fold_loss)

    ## cross validation value
    cross_validation_value = sum(folds_losses) / folds_number

    if verbose: print(f'>> final cross validation value: {cross_validation_value}')

    return cross_validation_value
