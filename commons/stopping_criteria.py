from __future__ import annotations
import numpy as np
from typing import Any, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from predictors.treepredictors import TreePredictor, TreePredictorNode # circular import

## Stopping conditions

class StopCondition(ABC):
    """Abstract base class representing conditional statements appliable to stop
    the growing of tree predictors.
    Stop conditions can be implemented in different ways and act either on the
    whole tree structure or on single nodes of the tree, so the test method
    signature may differ but all of them must return boolean values indicating
    whether to stop (True) or not (False).
    """

    @abstractmethod
    def stop(self, object:Any) -> bool:
        """Tests the given object status according to the underlying conditional
        statement represented by this calss and its parameters, returning a
        Boolean value indicating whether tree growing should be stopped or not.

        Args:
            object (Any): object to test.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        raise NotImplemented

    def __str__(self) -> str:
        return '<generic stop condition>'

## Tree stopping conditions

class TreeStopCondition(StopCondition):
    """Abstract class extending the base StopCondition abstract class to represent
    conditional statements specifically designed to test tree structures.
    """

    @abstractmethod
    def stop(self, tree:TreePredictor) -> bool:
        """Tests the given tree status according to the underlying conditional
        statement represented by this calss and its parameters, returning a
        Boolean value indicating whether tree growing should be stopped or not.

        Args:
            tree (TreePredictor): tree to test.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        raise NotImplemented

    def __str__(self) -> str:
        return '<tree stop condition>'
    
class TreeMaximumDepth(TreeStopCondition):
    """Instances of this class represent stop condition statements producing a
    decision about the stop of a tree growing process depending on the depth
    reached by the tree.
    """

    def __init__(self, maximum_depth:int) -> None:
        """Initializes a new TreeMaximumDepth instance by specifiying the maximum
        depth that tested tree can reach.

        Args:
            maximum_depth (int): maximum depth to use in checks.
        """

        super().__init__()

        self._maximum_depth = maximum_depth

    def stop(self, tree:TreePredictor) -> bool:
        """Tests the given tree depth against the fixed maximum depth of this
        condition, returning a Boolean value indicating whether tree growing
        should be stopped or not.

        Args:
            tree (TreePredictor): tree to test.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        return tree._depth > self._maximum_depth
    
    def __str__(self) -> str:
        return f'tree depth > {self._maximum_depth}'
    
class TreeMaximumLeaves(TreeStopCondition):
    """Instances of this class represent stop condition statements producing a
    decision about the stop of a tree growing process depending on the number of
    leaf nodes reached by the tree.
    """

    def __init__(self, maximum_leaves:int) -> None:
        """Initializes a new TreeMaximumLeaves instance by specifiying the maximum
        number of leaf nodes that tested tree can reach.

        Args:
            maximum_leaves (int): maximum number of leaves to use in checks.
        """

        super().__init__()

        self._maximum_leaves = maximum_leaves

    def stop(self, tree:TreePredictor) -> bool:
        """Tests the given tree number of leaves against the fixed maximum number
        of this condition, returning a Boolean value indicating whether tree
        growing should be stopped or not.

        Args:
            tree (TreePredictor): tree to test.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        return tree._leaves_count > self._maximum_leaves
    
    def __str__(self) -> str:
        return f'tree leaves count > {self._maximum_leaves}'
    
class TreeMaximumNodes(TreeStopCondition):
    """Instances of this class represent stop condition statements producing a
    decision about the stop of a tree growing process depending on the number of
    total nodes reached by the tree.
    """

    def __init__(self, maximum_nodes:int) -> None:
        """Initializes a new TreeMaximumNodes instance by specifiying the maximum
        number of nodes that tested tree can reach.

        Args:
            maximum_nodes (int): maximum number of nodes to use in checks.
        """

        super().__init__()

        self._maximum_nodes = maximum_nodes

    def stop(self, tree:TreePredictor) -> bool:
        """Tests the given tree number of nodes against the fixed maximum number
        of this condition, returning a Boolean value indicating whether tree
        growing should be stopped or not.

        Args:
            tree (TreePredictor): tree to test.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        return tree._nodes_count > self._maximum_nodes
    
    def __str__(self) -> str:
        return f'tree nodes count > {self._maximum_nodes}'

## Node stopping criteria

class NodeStopCondition(StopCondition):
    """Abstract class extending the base StopCondition abstract class to represent
    conditional statements specifically designed to test tree nodes and their samples.
    """

    @abstractmethod
    def stop(self, node:TreePredictorNode, labels:np.ndarray) -> bool:
        """Tests the given node status and label values according to the
        underlying conditional statement represented by this calss and its
        parameters, returning a Boolean value indicating whether tree growing
        should be stopped or not.

        Args:
            node (TreePredictorNode): node to test.
            labels (np.ndarray): labels associated to the node.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        raise NotImplemented

    def __str__(self) -> str:
        return f'<node stop condition>'
    
class NodeMinimumSamples(NodeStopCondition):
    """Instances of this class represent stop condition statements producing a
    decision about the stop of a tree growing process depending on the number
    of samples that are associated to the node.
    """

    def __init__(self, minimum_samples:int) -> None:
        """Initializes a new NodeMinimumSamples instance by specifiying the minimum
        number of samples that a tested node can have.

        Args:
            minimum_samples (int): minimum number of samples to use in checks.
        """

        super().__init__()

        self._minimum_samples = minimum_samples

    def stop(self, _:TreePredictorNode, labels:np.ndarray) -> bool:
        """Tests the number of given samples associated to the given node against
        the fixed minimum number of this condition, returning a Boolean value
        indicating whether tree growing should be stopped or not.

        Args:
            _ (TreePredictorNode): node to test (not used).
            labels (np.ndarray): labels associated to the node.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        return len(labels) <= self._minimum_samples
    
    def __str__(self) -> str:
        return f'samples number <= {self._minimum_samples}'

class NodeImpurityLevel(NodeStopCondition):
    """Instances of this class represent stop condition statements producing a
    decision about the stop of a tree growing process depending on the impurity
    level (according to a specified measurement) of samples that are associated
    to the node.
    """

    def __init__(self, measurement:Callable[[np.ndarray], np.number], threshold:np.number) -> None:
        """Initializes a new NodeImpurityLevel instance by specifiying the
        function to use for measuring sample impurity and the threshold impurity
        value that the samples of a tested node can have.

        Args:
            measurement (Callable[[np.ndarray], np.number]): measurement of impurity.
            threshold (np.number): threshold value to use in checks.
        """

        super().__init__()

        self._measurement = measurement
        self._threshold = threshold

    def stop(self, _:TreePredictorNode, labels:np.ndarray) -> bool:
        """Tests the impurity level of given samples associated to the given
        node against the fixed threshold of this condition, returning a Boolean
        value indicating whether tree growing should be stopped or not.

        Args:
            _ (TreePredictorNode): node to test (not used).
            labels (np.ndarray): labels associated to the node.

        Returns:
            bool: True if the growing process should be stopped, otherwise False
            if the growing process could continue.
        """

        return self._measurement(labels) <= self._threshold

    def __str__(self) -> str:
        return f'{self._measurement.__name__} impurity <= {self._threshold}'
