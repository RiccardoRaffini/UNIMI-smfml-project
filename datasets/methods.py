import numpy as np
import pandas as pd
from typing import Any, Callable

def dataset_information(dataset:pd.DataFrame, normalize:bool = False) -> dict[str, Any]:
    """Returns some useful information about the given dataset. Numerical
    information can be normalized with respect to the size of the dataset.

    Args:
        dataset (pd.DataFrame): dataframe representing the dataset from which to
        obtaine information.
        normalize (bool, optional): If True, divides the numerical information
        by the number of examples in dataset, otherwise original values are
        returned. Defaults to False.

    Returns:
        dict[str, Any]: fixed-keys dictionary containing the information
        extracted from the dataset.
    """

    information = dict()

    information['samples'] = dataset.shape[0]
    information['features'] = dataset.shape[1]
    information['features names'] = np.array(dataset.columns)
    information['classes'] = np.unique(dataset['class'])
    information['classes samples'] = dataset['class'].value_counts()
    information['null samples'] = dataset.isnull().any(axis=1).sum()
    information['null features'] = dataset.isnull().sum(axis=0)

    if normalize:
        information['classes samples'] /= information['samples']
        information['null samples'] /= information['samples']
        information['null features'] /= information['samples']

    return information

def delete_dataset_features(dataset:pd.DataFrame, threshold:np.number) -> pd.DataFrame:
    """Applies deletion technique to features in the given dataset, removing
    features whose null (NA/NaN) values count is above the specified threshold value.

    Args:
        dataset (pd.DataFrame): dataframe representing the dataset to process.
        threshold (np.number): percentage of samples to use as threshold for
        discarding a feature.

    Returns:
        pd.DataFrame: processed dataset containing only features with null values
        count under the threshold.
    """

    samples_threshold = dataset.shape[0] * threshold
    null_columns = np.array([column_name for column_name in dataset.columns if dataset[column_name].isnull().sum() >= samples_threshold])

    new_dataset = dataset.drop(null_columns, axis=1)

    return new_dataset

def fill_dataset_samples(dataset:pd.DataFrame, filling_value:Any) -> pd.DataFrame:
    """Applies imputation technique to the samples in the given dataset, assigning
    the specified filling value to all their missing values (NA/NaN).

    Args:
        dataset (pd.DataFrame): dataframe representing the dataset to process.
        filling_value (Any): any compatiblevalue usable to replace the missing value.

    Returns:
        pd.DataFrame: processed dataset not containing null values.
    """

    new_dataset = dataset.fillna(filling_value)

    return new_dataset

def extract_samples_labels(dataset:pd.DataFrame, labels_map:Callable[[Any], Any] = lambda x: x) -> tuple[np.ndarray, np.ndarray]:
    """Separates labels from samples data contained in the given dataset. It
    also possible to apply a vectorized transformation to the extracted labels.

    Args:
        dataset (pd.DataFrame): dataframe representing the dataset containing
        samples and labels.
        labels_map (Any, optional): vectorized function applied to the extracted
        labels. Defaults to lambda x: x.

    Returns:
        tuple[np.ndarray, np.ndarray]: pair of samples and labels.
    """

    # labels extraction
    labels = dataset['class'].to_numpy()
    labels = np.apply_along_axis(labels_map, 0, labels)

    # samples extraction
    samples = dataset.drop('class', axis=1).to_numpy()

    return samples, labels
