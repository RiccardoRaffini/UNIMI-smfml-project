import numpy as np
from typing import Any, Callable

class TreePredictorNode:
    """Instances of this calss can be used to represent either internal or leaf
    nodes of a tree predictor.

    The difference between an internal node and a leaf node is given by the
    available attribute of the node. A leaf node is characterized by an
    assigned label, whereas an internal node is characterized by a decision
    test defined on a samples feature and eventually by some children nodes.

    According to their type, tree nodes allow to apply a decision test to
    determine the next node to use for prediction or to determine a final
    prediction label.
    """

    def __init__(self, feature_index:int = None, decision_test:Callable[[np.ndarray], int] = None, label:Any = None, children_nodes:list['TreePredictorNode'] = None) -> None:
        """Initializes a new TreePredictorNode instance by specifying which
        attributes it should hold and consequently its type (internal or leaf).

        Args:
            feature_index (int, optional): index of the sample feature on which
            the decision test should be applied. Defaults to None if this is a
            leaf node.
            decision_test (Callable[[np.ndarray], int], optional): function
            performing the test on the indexed feature and returning the next
            node index. Defaults to None if this is a leaf node.
            label (Any, optional): final prediction label of the node. Defaults
            to None if this an internal node.
            children_nodes (list[&#39;TreePredictorNode&#39;], optional): list
            of tree predictor nodes that represent the children of this node.
            Defaults to None if this is a leaf node or it does not have children.
        """

        self.feature_index = feature_index
        self.decision_test = decision_test
        self.label = label
        self.children = [] if children_nodes is None else children_nodes

    def is_leaf(self) -> bool:
        """Returns whether this node is a leaf or not.

        Returns:
            bool: True if this node is a leaf, False otherwise.
        """

        return self.label is not None

    def test(self, sample:np.ndarray) -> None | int:
        """Applies the decision test associated to this node to the provided
        sample, specifically to the feature indexed by the value assigned to
        this node.
        The returned decision corresponds to the index of a child node of this
        node that should be applied for the computation step.
        The decision is None if it is called on a leaf node, not supposed to
        have a decision test.

        Args:
            sample (np.ndarray): sample to which apply the decision test.

        Returns:
            None | int: the index of the next child node or None if this node
            is a leaf.
        """

        if self.is_leaf():
            return None
        
        feature_value = sample[self.feature_index]
        decision = self.decision_test(feature_value)

        return decision
    
    def __str__(self) -> str:
        if self.is_leaf():
            result = f'| Leaf node > label: {self.label} |'
        else:
            result = f'{{ Internal node > feature index: {self.feature_index}, decision test: {self.decision_test}, children number: {len(self.children)} }}'

        return result
