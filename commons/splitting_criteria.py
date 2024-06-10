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
