import numpy as np
import math
from typing import Callable
from typing import Any, Type, Tuple, Dict
from sklearn.model_selection import train_test_split

from predictors.treepredictors import TreePredictor
from commons.plotting import format_hyperparameters

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
    """Splits the given data collection in the provided number of folds.
    The number of elements in each folder is the ceiling approximation 'm/k' of
    the number of elements 'm' in the collection and the number of folds 'k',
    thus if 'm' is not a multiple of 'k', the last fold will contain just 'm%k'
    elements.

    Args:
        data (np.ndarray): collection of elements to split in folds.
        folds_number (int): number of folds.

    Returns:
        np.ndarray[np.ndarray]: collection of folds containing split elements.
    """

    data_size = len(data)
    fold_size = math.ceil(data_size / folds_number)

    data_folds = [data[i*fold_size:(i+1)*fold_size] for i in range(folds_number)]
    data_folds = np.array(data_folds)

    return data_folds

def cross_validation(model:Type, loss:Callable[[np.ndarray, np.ndarray], np.ndarray], data_folds:np.ndarray[np.ndarray], labels_folds:np.ndarray[np.ndarray], verbose:bool = False) -> np.number:
    """Computes the cross validation value of the provided data (and labels)
    folds, each of one is evaluated according to the loss, using a model trained
    with the other folds.

    Args:
        model (TreePredictor): type of predictive model to train on the folds.
        loss (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
        to use for evaluating the model on folds data.
        data_folds (np.ndarray[np.ndarray]): collection of folds containing the samples.
        labels_folds (np.ndarray[np.ndarray]): collection of folds containing
        the labels associated to data.
        verbose (bool, optional): determines wheter or not to print information
        during the process. Defaults to False for no prints.

    Returns:
        np.number: cross validation value.
    """
    
    folds_number = len(data_folds)
    folds_losses = []

    if verbose: print(f'Running cross validation on {folds_number} folds of size {len(data_folds[0])}')

    ## Iterate on folds
    for k in range(folds_number):
        if verbose: print(f'Fold {k} iteration:')

        ## Split folds
        train_data = np.concatenate(data_folds[np.arange(folds_number) != k], axis=0)
        train_labels = np.concatenate(labels_folds[np.arange(folds_number) != k], axis=0)
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

def holdout_cross_validation(model:Type[TreePredictor], loss:Callable[[np.ndarray, np.ndarray], np.ndarray],
                             data:np.ndarray, labels:np.ndarray, splits_ratios:Tuple[int, int, int],
                             hyperparameters:np.ndarray[np.ndarray], shuffle:bool = True,
                             random_seed:int = None, verbose:bool = False) -> Tuple[np.number, Tuple]:
    """Executes the holdout cross validation on the provided collection of data
    and labels, splitting them in train/validation/test sets and specifically
    evaluating a TreePredictor predictor trained on all the hyperparameters in
    the given grid.
    This function only works on TreePredictor models, so also hyperparameter
    must be compatible.

    Args:
        model (Type[TreePredictor]): type of predictive model to train on the dataset.
        loss (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
        to use for evaluating the model on folds data.
        data (np.ndarray): collection of samples.
        labels (np.ndarray): collection of labels associated to samples.
        splits_ratios (Tuple[int, int, int]): ratios for splitting train, validation
        and test sets.
        hyperparameters (np.ndarray[np.ndarray]): linearized grid of possible
        hyperparameters to use in the inner cross validation procedure.
        shuffle (bool, optional): determines whether or not to shuffle data in
        the given set. Defaults to True for shuffling them.
        random_seed (int, optional): random seed to use for shuffling data.
        Defaults to None for not using a seed.
        verbose (bool, optional): determines wheter or not to print information
        during the process. Defaults to False for no prints.

    Returns:
        Tuple[np.number, Tuple]: holdout validation value and the hyperparameters
        providing the best result.
    """
    
    ## Create splits
    train_ratio, validation_ratio, test_ratio = splits_ratios
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=1-train_ratio, shuffle=shuffle, random_state=random_seed)
    validation_data, test_data, validation_labels, test_labels = train_test_split(test_data, test_labels, test_size=test_ratio/(validation_ratio+test_ratio))

    if verbose: print(f'split {len(data)} data into train [{len(train_data)}], validation [{len(validation_data)}] and test [{len(test_data)}]')

    ## Iterating on hyperparameters
    hyperparameters_losses = []
    for i, hyperparameter in enumerate(hyperparameters):
        if verbose: print(f'- cross validation using parameters [{i}] {format_hyperparameters(hyperparameter)}')

        ## Define parameters
        continuous_condition = hyperparameter[0]
        categorical_condition = hyperparameter[1]
        decision_metric = hyperparameter[2]
        tree_stopping_criteria = hyperparameter[3]
        node_stopping_criteria = hyperparameter[4]

        ## Define model
        current_model = model(continuous_condition, categorical_condition, decision_metric, tree_stopping_criteria, node_stopping_criteria)

        ## Train model on train set
        if verbose: print(f'Training model...')

        current_model.fit(train_data, train_labels)

        ## Evaluate model on validation set
        if verbose: print(f'Validating model...')

        hyperparameter_loss = samples_error(current_model.predict, loss, validation_labels, validation_data)
        hyperparameters_losses.append(hyperparameter_loss)

        if verbose: print(f'Validation loss: {hyperparameter_loss}')

    ## cross validation value
    holdout_cross_validation_value = np.min(hyperparameters_losses)

    if verbose: print(f'>> Final holdout cross validation error: {holdout_cross_validation_value}')

    ## Find best hyperparameters
    best_index = np.argmin(hyperparameters_losses)
    best_hyperparameters = hyperparameters[best_index]

    if verbose: print(f'>> Final best hyperparameters: {format_hyperparameters(best_hyperparameters)}')

    return holdout_cross_validation_value, best_hyperparameters

def nested_cross_validation(model:Type[TreePredictor], loss:Callable[[np.ndarray, np.ndarray], np.ndarray],
                              data:np.ndarray, labels:np.ndarray,
                              folds_number:int, hyperparameters:np.ndarray[np.ndarray],
                              verbose:bool = False) -> Tuple[np.number, Tuple]:
    """Executes the nested cross validation on the provided collection of data
    and labels, splitting them in the specified number of folds and specifically
    evaluating a TreePredictor predictor trained on all the hyperparameters in
    the given grid.
    This function only works on TreePredictor models, so also hyperparameter
    must be compatible.

    Args:
        model (Type[TreePredictor]): type of predictive model to train on the folds.
        loss (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
        to use for evaluating the model on folds data.
        data (np.ndarray): collection of samples.
        labels (np.ndarray): collection of labels associated to samples.
        folds_number (int): number of folds to create.
        hyperparameters (np.ndarray[np.ndarray]): linearized grid of possible
        hyperparameters to use in the inner cross validation procedure.
        verbose (bool, optional): determines wheter or not to print information
        during the process. Defaults to False for no prints.

    Returns:
        Tuple[np.number, Tuple]: cross validation value and the hyperparameters
        providing the best result.
    """
    
    ## Create folds
    data_folds = create_folds(data, folds_number)
    labels_folds = create_folds(labels, folds_number)

    if verbose: print(f'Running nested cross validation on {folds_number} folds of size {len(data_folds[0])}')

    ## Iterate on folds
    folds_best_losses = []
    folds_best_hyperparameters = []
    for k in range(folds_number):
        if verbose: print(f'Fold [{k}] iteration:')
    
        fold_cross_validation_values = []

        ## Split folds
        train_data_folds = data_folds[np.arange(folds_number) != k]
        train_data = np.concatenate(train_data_folds, axis=0)
        train_labels_folds = labels_folds[np.arange(folds_number) != k]
        train_labels = np.concatenate(train_labels_folds, axis=0)
        test_data = data_folds[k]
        test_labels = labels_folds[k]

        ## Iterating on hyperparameters
        if verbose: print(f'Nested cross validation for folds [-{k}]:')

        for hyperparameter in hyperparameters:
            if verbose: print(f'- cross validation using parameters {hyperparameter}')

            ## Define parameters
            continuous_condition = hyperparameter[0]
            categorical_condition = hyperparameter[1]
            decision_metric = hyperparameter[2]
            tree_stopping_criteria = hyperparameter[3]
            node_stopping_criteria = hyperparameter[4]

            ## Define model
            current_model = model(continuous_condition, categorical_condition, decision_metric, tree_stopping_criteria, node_stopping_criteria)

            ## cross validation
            cross_validation_value = cross_validation(current_model, loss, train_data_folds, train_labels_folds)
            fold_cross_validation_values.append(cross_validation_value)

            if verbose: print(f'  result: {cross_validation_value}')

        ## Find best hyperparameters
        best_index = np.argmin(fold_cross_validation_values)
        fold_best_hyperparameters = hyperparameters[best_index]
        folds_best_hyperparameters.append(fold_best_hyperparameters)

        if verbose: print(f'Folds [-{k}] best hyperparameters: {fold_best_hyperparameters}')

        ## Define best model
        continuous_condition = fold_best_hyperparameters[0]
        categorical_condition = fold_best_hyperparameters[1]
        decision_metric = fold_best_hyperparameters[2]
        tree_stopping_criteria = fold_best_hyperparameters[3]
        node_stopping_criteria = fold_best_hyperparameters[4]

        best_model = model(continuous_condition, categorical_condition, decision_metric, tree_stopping_criteria, node_stopping_criteria)

        ## Train best model
        if verbose: print(f'Training folds [-{k}] best model...')

        best_model.fit(train_data, train_labels)

        ## Test best model
        if verbose: print(f'Testing fold [{k}] best model...')

        fold_best_loss = samples_error(best_model.predict, loss, test_labels, test_data)
        folds_best_losses.append(fold_best_loss)

        if verbose: print(f'> Fold [{k}] loss: {fold_best_loss}')

    ## Average error
    nested_cross_validation_value = np.sum(folds_best_losses) / folds_number

    if verbose: print(f'>> Final nested cross validation error: {nested_cross_validation_value}')

    ## Best hyperparameters
    best_index = np.argmin(folds_best_losses)
    best_hyperparameters = folds_best_hyperparameters[best_index]

    if verbose: print(f'>> Final best hyperparameters: {best_hyperparameters}')

    return nested_cross_validation_value, best_hyperparameters

def flat_cross_validation(model:Type[TreePredictor], loss:Callable[[np.ndarray, np.ndarray], np.ndarray],
                              data:np.ndarray, labels:np.ndarray,
                              folds_number:int, hyperparameters:np.ndarray[np.ndarray],
                              verbose:bool = False) -> Tuple[np.number, Tuple]:
    """Executes the flat cross validation variant on the provided collection of
    data and labels, splitting them in the specified number of folds and
    specifically evaluating a TreePredictor predictor trained on all the
    hyperparameters in the given grid.
    This function only works on TreePredictor models, so also hyperparameter
    must be compatible.

    Args:
        model (Type[TreePredictor]): type of predictive model to train on the folds.
        loss (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
        to use for evaluating the model on folds data.
        data (np.ndarray): collection of samples.
        labels (np.ndarray): collections of labels associated to samples.
        folds_number (int): number of folds to create.
        hyperparameters (np.ndarray[np.ndarray]): linearized grid of possible
        hyperparameters to use.
        verbose (bool, optional): determines wheter or not to print information
        during the process. Defaults to False for no prints.

    Returns:
        Tuple[np.number, Tuple]: flat cross validation value and the hyperparameters
        returning this value.
    """

    ## Create folds
    data_folds = create_folds(data, folds_number)
    labels_folds = create_folds(labels, folds_number)

    if verbose: print(f'Running flat cross validation on {folds_number} folds of size {len(data_folds[0])}')

    ## Iterating on hyperparameters
    hyperparameters_losses = []
    for i, hyperparameter in enumerate(hyperparameters):
        if verbose: print(f'- cross validation using parameters [{i}] {hyperparameter}')

        ## Iterating on folds
        folds_losses = []
        for k in range(folds_number):
            if verbose: print(f'Fold [{k}] iteration:')

            ## Split folds
            train_data_folds = data_folds[np.arange(folds_number) != k]
            train_data = np.concatenate(train_data_folds, axis=0)
            train_labels_folds = labels_folds[np.arange(folds_number) != k]
            train_labels = np.concatenate(train_labels_folds, axis=0)
            test_data = data_folds[k]
            test_labels = labels_folds[k]

            ## Define parameters
            continuous_condition = hyperparameter[0]
            categorical_condition = hyperparameter[1]
            decision_metric = hyperparameter[2]
            tree_stopping_criteria = hyperparameter[3]
            node_stopping_criteria = hyperparameter[4]

            ## Define model
            current_model = model(continuous_condition, categorical_condition, decision_metric, tree_stopping_criteria, node_stopping_criteria)

            ## Train model
            if verbose: print(f'Training fold [-{k}] best model...')

            current_model.fit(train_data, train_labels)

            ## Test model
            if verbose: print(f'Testing fold [{k}] best model...')

            fold_loss = samples_error(current_model.predict, loss, test_labels, test_data)
            folds_losses.append(fold_loss)

            if verbose: print(f'result: {fold_loss}')

        hyperparameter_loss = np.sum(folds_losses)
        hyperparameters_losses.append(hyperparameter_loss)

        if verbose: print(f'> Hyperparameter total loss: {hyperparameter_loss}')

    ## Best hyperparameters
    best_index = np.argmin(hyperparameters_losses)
    best_hyperparameters = hyperparameters[best_index]

    if verbose: print(f'>> Final best hyperparameters: {best_hyperparameters}')

    ## Average loss
    flat_cross_validation_value = hyperparameters_losses[best_index] / folds_number

    if verbose: print(f'>> Final flat cross validation error: {flat_cross_validation_value}')

    return flat_cross_validation_value, best_hyperparameters
