import math
import numpy as np
from typing import Callable, Any
from abc import ABC, abstractmethod

## Test conditions

class Condition(ABC):
    """Abstract base class representing conditional statements appliable as
    decision test inside internal nodes of tree predictors.
    Decision can be implemented in different ways according to the statements
    they express, but they all must provide a test method for comparing a given
    value and returning a partition index.
    """

    @abstractmethod
    def test(self, value:Any) -> int:
        """Tests the given value according to the underlying conditional
        statement represented by this class and its parameters, returning the
        index of the partition where value belongs.

        Args:
            value (Any): value to compare.

        Returns:
            int: index of the partition resulting from the test.
        """

        raise NotImplementedError

    def __str__(self) -> str:
        return '<generic condition>'

class BinaryCondition(Condition):
    """Abstract class extending the base Condition abstract class to represent
    conditional statement limited to only two outcomes (binary conditions).
    """

    @abstractmethod
    def test(self, value:Any) -> bool:
        """Tests the given value according to the underlying conditional
        statement represented by this class and its parameters, returning the
        index of the partition where value belongs. Since there are only two
        possible outcomes, they are mapped the to Boolean values True and False.

        Args:
            value (Any): value to compare.

        Returns:
            bool: False (equivalent to 0) if test results in first partition,
            otherwise True (equivalent t0 1) if test results in second partition.
        """

        raise NotImplementedError

    def __str__(self) -> str:
        return '<binary condition>'

class ThresholdCondition(BinaryCondition):
    """Instances of this class represent conditional statements producing binary
    results according to a threshold test on numerical values.
    Statements are characterized by a fixed threshold and check if the provided
    values are smaller than or equal to it.
    """

    def __init__(self, threshold:np.number) -> None:
        """Initializes a new ThresholdCondition instance by specifying a threshold
        value to use in tests.

        Args:
            threshold (np.number): thresholding value to use in tests.
        """

        super().__init__()

        self._threshold = threshold

    def test(self, value:np.number) -> bool:
        """Tests the given value againist the fixed threshold of this condition,
        returning the index of the partition where value belongs.

        Args:
            value (np.number): value to compare.

        Returns:
            bool: False (equivalent to 0) if test results in first partition,
            otherwise True (equivalent t0 1) if test results in second partition.
        """

        return value <= self._threshold
    
    def __str__(self) -> str:
        return f'x <= {self._threshold}'
    
class MembershipCondition(BinaryCondition):
    """Instances of this class represent conditional statements producing binary
    results according to a membership test on any kind of values.
    Statements are characterized by a reference set of member values and check
    if the provided values belong to the set.
    """

    def __init__(self, members:set[Any] | Any) -> None:
        """Initializes a new MembershipCondition instance by specifying a
        reference member or a reference set of members to use in tests.

        Args:
            members (set[Any] | Any): single value of set of values representing
            the reference set.
        """

        super().__init__()

        self._members = members if type(members) == set else {members}

    def test(self, value: Any) -> bool:
        """Tests if the given value is inside the reference set of members of
        this condition, returning the index of the partition where value belongs.

        Args:
            value (Any): value to compare.

        Returns:
            bool: False (equivalent to 0) if test results in first partition,
            otherwise True (equivalent t0 1) if test results in second partition.
        """

        return value in self._members
    
    def __str__(self) -> str:
        return f'x in {self._members}'

class EqualityCondition(BinaryCondition):
    """Instances of this class represent conditional statements producing binary
    results according to an equality test on any kind of values.
    Statements are characterized by a reference value and check if the provided
    values are equal to it.
    """

    def __init__(self, reference_value:Any) -> None:
        """Initializes a new EqualityCondition instance by specifying a reference
        value to use in tests.

        Args:
            reference_value (Any): reference value to use in tests.
        """

        super().__init__()

        self._reference_value = reference_value

    def test(self, value: Any) -> bool:
        """Tests the given value againist the fixed reference value of this
        condition, returning the index of the partition where value belongs.

        Args:
            value (Any): value to compare.

        Returns:
            bool: False (equivalent to 0) if test results in first partition,
            otherwise True (equivalent t0 1) if test results in second partition.
        """

        return value == self._reference_value
    
    def __str__(self) -> str:
        return f'x == {self._reference_value}'

## Impurity indices

def entropy(values:np.ndarray) -> np.number:
    """Returns the entropy of the provided collection of values.

    Args:
        values (np.ndarray): collection of values for which to compute entropy.

    Returns:
        np.number: entropy of the values.
    """

    _, values_count = np.unique(values, return_counts=True)
    values_probabilities = values_count / values.size

    entropy_value = - np.sum([probability * np.log2(probability) for probability in values_probabilities if probability > 0])

    return entropy_value

def gini_index(values:np.ndarray) -> np.number:
    """Returns the Gini index of the provided collection of values.

    Args:
        values (np.ndarray): collection of values for which to compute Gini index.

    Returns:
        np.number: Gini index of the values.
    """

    _, values_count = np.unique(values, return_counts=True)
    values_probabilities = values_count / values.size

    gini_value = np.sum([value_probability ** 2 for value_probability in values_probabilities])

    return gini_value

def gini_impurity(values:np.ndarray) -> np.number:
    """Returns the Gini impurity of the provided collection of values.

    Args:
        values (np.ndarray): collection of values for which to compute Gini impurity.

    Returns:
        np.number: Gini impurity of the values.
    """

    gini_value = gini_index(values)
    gini_impurity_value = 1 - gini_value

    return gini_impurity_value

def minimum(values:np.ndarray) -> np.number:
    """Returns the minimum absolute frequency of the unique values in the
    provided collection of values.

    Args:
        values (np.ndarray): ollection of values for which to compute Gini minimum.

    Returns:
        np.number: minimum of the values.
    """

    _, values_count = np.unique(values, return_counts=True)

    minimum_value = np.min(values_count)

    return minimum_value

## Measures gain

def measure_gain(measure:Callable[[np.ndarray], np.number], feature_values:np.ndarray, labels:np.ndarray, decision:Callable[[Any], int]) -> np.number:
    """Computes the mesurement gain based on the given measure, simulating the
    split of the provided features values and associated labels according to the
    decision criterium.
    The gain is computed as the difference between the initial measurement
    before the split and the weighted sum of the measurements in each split.

    Args:
        measure (Callable[[np.ndarray], np.number]): measurement function to use
        in computation.
        feature_values (np.ndarray): collection of values to which apply the computation.
        labels (np.ndarray): labels associated to the feature values.
        decision (Callable[[Any], int]): decision criterium to apply for
        splitting the feature values.

    Returns:
        np.number: measurement gain of the split.
    """

    ## initial measurement
    initial_measure = measure(labels)

    ## splits
    decision_results = np.apply_along_axis(np.vectorize(decision), 0, feature_values)
    decision_unique_values = np.unique(decision_results)
    splits_labels = [labels[np.where(decision_results == unique_value)] for unique_value in decision_unique_values]

    ## splits measure
    feature_size = feature_values.size
    splits_sizes = [split_labels.size for split_labels in splits_labels]

    splits_measure = 0
    for i in range(len(splits_labels)):
        split_weight = splits_sizes[i] / feature_size
        split_measure = measure(splits_labels[i])
        splits_measure += split_weight * split_measure

    ## measurement gain
    final_gain = initial_measure - splits_measure

    return final_gain

def information_gain(feature_values:np.ndarray, labels:np.ndarray, decision:Callable[[Any], int]) -> np.number:
    """Computes the information gain, simulating the split of the provided
    features values and associated labels according to the decision criterium.
    The gain is computed as the difference between the initial entropy before
    the split and the weighted sum of the entropies in each split.

    Args:
        feature_values (np.ndarray): collection of values to which apply the computation.
        labels (np.ndarray): labels associated to the feature values.
        decision (Callable[[Any], int]): decision criterium to apply for
        splitting the feature values.

    Returns:
        np.number: information gain of the split.
    """

    gain = measure_gain(entropy, feature_values, labels, decision)

    return gain

def gini_impurity_gain(feature_values:np.ndarray, labels:np.ndarray, decision:Callable[[Any], int]) -> np.number:
    """Computes the gini impurity gain, simulating the split of the provided
    features values and associated labels according to the decision criterium.
    The gain is computed as the difference between the initial gini impurity
    before the split and the weighted sum of the gini impurities in each split.

    Args:
        feature_values (np.ndarray): collection of values to which apply the computation.
        labels (np.ndarray): labels associated to the feature values.
        decision (Callable[[Any], int]): decision criterium to apply for
        splitting the feature values.

    Returns:
        np.number: gini impurity gain of the split.
    """

    gain = measure_gain(gini_impurity, feature_values, labels, decision)

    return gain

def misclassification_gain(feature_values:np.ndarray, labels:np.ndarray, decision:Callable[[Any], int]) -> np.number:
    """Computes the classification error gain, simulating the split of the
    provided features values and associated labels according to the decision criterium.
    The gain is computed as the difference between the initial minimum error
    before the split and the weighted sum of the minimum errors in each split.

    Args:
        feature_values (np.ndarray): collection of values to which apply the computation.
        labels (np.ndarray): labels associated to the feature values.
        decision (Callable[[Any], int]): decision criterium to apply for
        splitting the feature values.

    Returns:
        np.number: misclassification gain of the split.
    """

    gain = measure_gain(minimum, feature_values, labels, decision)

    return gain

## Other measures

def chi_square(feature_values:np.ndarray, labels:np.ndarray, decision:Callable[[Any], int]) -> np.number:
    """Computes the Chi-squared metric, simulating the split of the provided
    features values and associated labels according to the decision criterium.
    The Chi-square of the decision is computed as the sum of the Chi-squares of
    each split, in turn computed as the sum of Chi-squares of each unique label.

    Args:
        feature_values (np.ndarray): collection of values to which apply the computation.
        labels (np.ndarray): labels associated to the feature values.
        decision (Callable[[Any], int]): decision criterium to apply for
        splitting the feature values.

    Returns:
        np.number: Chi-square value of the split.
    """

    ## splits
    decision_results = np.apply_along_axis(np.vectorize(decision), 0, feature_values)
    decision_unique_values = np.unique(decision_results)
    splits_labels = [labels[np.where(decision_results == unique_value)] for unique_value in decision_unique_values]

    ## Expected values distribution (parent)
    total_labels = len(labels)
    unique_labels, labels_counts = np.unique(labels, return_counts=True)
    expected_frequency = {unique_label: label_count/total_labels for unique_label, label_count in zip(unique_labels, labels_counts)}

    ## Splits Chi-square
    splits_chi_squares = []
    for i in range(len(splits_labels)):
        split_labels = splits_labels[i]
        unique_labels, labels_counts = np.unique(split_labels, return_counts=True)

        ## Expected values and actual values (child)
        expected_values = dict()
        actual_values = dict()
        for unique_label, label_count in zip(unique_labels, labels_counts):
            expected_values[unique_label] = label_count * expected_frequency[unique_label]
            actual_values [unique_label] =  label_count

        ## Labels Chi-squares
        labels_chi_squares = [math.sqrt((actual_values[label] - expected_values[label])**2 / expected_values[label]) for label in actual_values.keys()]

        ## Split Chi-square
        split_chi_square = np.sum(labels_chi_squares)
        splits_chi_squares.append(split_chi_square)

    ## Decision Chi-square
    decision_chi_square = np.sum(splits_chi_squares)

    return decision_chi_square
