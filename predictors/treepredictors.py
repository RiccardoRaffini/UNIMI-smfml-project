import numpy as np
from typing import Any, Callable, Type
from collections import Counter

from commons.splitting_criteria import Condition
from commons.splitting_criteria import information_gain, entropy
from commons.stopping_criteria import TreeStopCondition, TreeMaximumDepth, NodeStopCondition, NodeImpurityLevel

class TreePredictorNode:
    """Instances of this calss can be used to represent either internal or leaf
    nodes of a tree predictor.

    The difference between an internal node and a leaf node is given by the
    available attribute of the node. A leaf node is characterized by an
    assigned label, whereas an internal node is characterized by a condition
    test defined on a samples feature and eventually by some children nodes.

    According to their type, tree nodes allow to apply the condition test to
    determine the next node to use for prediction or to determine a final
    prediction label.
    """

    def __init__(self, feature_index:int = None, condition:Condition = None, label:Any = None, children_nodes:list['TreePredictorNode'] = None) -> None:
        """Initializes a new TreePredictorNode instance by specifying which
        attributes it should have and consequently its type (internal or leaf).

        Args:
            feature_index (int, optional): index of the sample feature on which
            the condition test should be applied. Defaults to None if this is a
            leaf node.
            condition (Condition, optional): conditional object usable to
            perform the test on the indexed feature and returning the next node
            index. Defaults to None if this is a leaf node.
            label (Any, optional): final prediction label of the node. Defaults
            to None if this an internal node.
            children_nodes (list[&#39;TreePredictorNode&#39;], optional): list
            of tree predictor nodes that represent the children of this node.
            Defaults to None if this is a leaf node or it does not have children.
        """

        self.feature_index = feature_index
        self.condition = condition
        self.label = label
        self.children = [] if children_nodes is None else children_nodes

    def is_leaf(self) -> bool:
        """Returns whether this node is a leaf or not.

        Returns:
            bool: True if this node is a leaf, False otherwise.
        """

        return self.label is not None

    def test(self, sample:np.ndarray) -> None | int:
        """Applies the condition test associated to this node to the provided
        sample, specifically to the feature indexed by the value assigned to
        this node.
        The returned value corresponds to the index of a child node of this
        node that should be applied in the next computation step.
        The value is None if it is called on a leaf node, not supposed to
        have a conditional test.

        Args:
            sample (np.ndarray): sample to which apply the condition test.

        Returns:
            None | int: the index of the next child node or None if this node
            is a leaf.
        """

        if self.is_leaf():
            return None
        
        feature_value = sample[self.feature_index]
        condition_value = self.condition.test(feature_value)

        return condition_value
    
    def __str__(self) -> str:
        if self.is_leaf():
            result = f'| Leaf node -- label: {self.label} |'
        else:
            result = f'{{ Internal node -- feature index: {self.feature_index}, condition test: {self.condition}, children number: {len(self.children)} }}'

        return result


class TreePredictor:
    """Instances of this class represent generic tree predictors, trainable to
    learn decision trees based on decision with a generic number of outcomes and
    appliable to any kind of features.
    Different aspects of the computation and training behavior can be customized
    to determine the kind of the obtained trees and/or their charcteristics.
    """

    def __init__(self,
        continuous_condition:Type[Condition], categorical_condition:Type[Condition],
        decision_metric:Callable[[np.ndarray, np.ndarray, Callable[[Any], int]], np.number] = None,
        tree_stopping_criteria:list[TreeStopCondition] = None,
        node_stopping_criteria:list[NodeStopCondition] = None
    ) -> None:
        """Initializes a new TreePredictor instance by specifying criteria
        required for its training and computation behavior.

        Args:
            continuous_condition (Type[Condition]): conditional statements type
            to use when internal node decision should be based on continuous feature.
            categorical_condition (Type[Condition]): conditional statement type
            to use when internal node decision should be based on categorical feature.
            decision_metric (Callable[[np.ndarray, np.ndarray, Callable[[Any], int]], np.number], optional):
            decision metric to use for evaluating possible decision of conditional
            statements. Defaults to None for using a base implementation of the
            information gain metric based on the entropy.
            tree_stopping_criteria (list[TreeStopCondition], optional): collection
            of stopping criteria for tree to check during training. Defaults to
            None for using only a default criteria limiting the depth of the
            tree to 100.
            node_stopping_criteria (list[NodeStopCondition], optional): collection
            of stopping criteria for nodes to check during training. Defaults to
            None for not using any criteria.
        """

        ## Tree structure
        self._root:TreePredictorNode = None
        self._depth = 0
        self._nodes_count = 0
        self._leaves_count = 0

        ## Computation criteria
        self._features_number:int = None
        self._continuous_condition = continuous_condition
        self._categorical_condition = categorical_condition
        self._decision_metric = decision_metric
        if self._decision_metric is None:
            self._decision_metric = information_gain

        ## Stopping criteria
        self._tree_stopping_criteria:list[TreeStopCondition] = []
        if tree_stopping_criteria is None:
            self._tree_stopping_criteria.append(TreeMaximumDepth(100))
        else:
            self._tree_stopping_criteria.extend(tree_stopping_criteria)
        
        self._node_stopping_criteria = [NodeImpurityLevel(entropy, 0)]
        if not node_stopping_criteria is None:
            self._node_stopping_criteria.extend(node_stopping_criteria)

    def fit(self, samples:np.ndarray, labels:np.ndarray, verbose:bool = False) -> None:
        """Trains the underlying model of this tree predictor using the given
        samples and their associated labels.

        Args:
            samples (np.ndarray): collection of samples usable to train the model.
            labels (np.ndarray): collection of labels associated to the samples.
            verbose (bool, optional): determines wheter or not to print
            information during the process. Defaults to False for no prints.
        """

        ## Information about sample
        _, samples_dimensionality = samples.shape
        self._features_number = samples_dimensionality

        ## New tree initialization
        common_label = self._common_label(labels)
        self._root = TreePredictorNode(label=common_label)
        self._depth = 1
        self._nodes_count = 1
        self._leaves_count = 1

        if verbose: print(f'initialized new tree')

        ## Tree building
        self._grow_tree(samples, labels, verbose)

    def predict(self, samples:np.ndarray) -> np.ndarray:
        """Predicts and retruns the labels of the given samples obtained using
        the underlying model of this tree predictor.

        Args:
            samples (np.ndarray): collection of samples to which predict labels.

        Returns:
            np.ndarray: prediction labels determined by the model. 
        """

        samples_predictions = np.array([self._traverse_tree(sample) for sample in samples])

        return samples_predictions

    def _grow_tree(self, samples:np.ndarray, labels:np.ndarray, verbose:bool = False) -> None:
        """Expands the current tree starting from its newly created root
        observing the provided samples and labels, determines when it is
        appropriated or not to expand nodes according to the stopping criteria
        associates to this tree, eventually finding the best splitting decision
        for intermediate nodes.

        Args:
            samples (np.ndarray): collections representing the samples usable
            to expand the tree.
            labels (np.ndarray): collection of labels associated to samples.
            verbose (bool, optional): determines wheter or not to print
            information during the process. Defaults to False for no prints.
        """

        if verbose: print(f'Expanding tree...')

        nodes_queue = [(self._root, np.arange(samples.shape[0]), self._depth)]

        while nodes_queue:
            if verbose: print(f'tree status > depth: {self._depth} total nodes: {self._nodes_count} leaf nodes: {self._leaves_count}\nnodes in queue: {len(nodes_queue)}')

            ## Current node data
            current_node, current_samples_indices, current_depth = nodes_queue.pop(0)
            current_node_samples = samples[current_samples_indices]
            current_node_labels = labels[current_samples_indices]

            if verbose: print(f'current node > label: {current_node.label} samples: {len(current_samples_indices)} depth: {current_depth}')

            ## Conditions check

            ### tree conditions
            if verbose: print(f'checking tree stop conditions:')

            stop = False
            for tree_stopping_criteria in self._tree_stopping_criteria:
                if verbose: print(f'- checking if {tree_stopping_criteria}')

                if tree_stopping_criteria.stop(self):
                    if verbose: print(f'>> stopping because {tree_stopping_criteria}')

                    stop = True
                    break

            if stop:
                continue
            
            ### node conditions
            if verbose: print(f'checking node stop conditions:')

            stop = False
            for node_stopping_criteria in self._node_stopping_criteria:
                if verbose: print(f'- checking if {node_stopping_criteria}')

                if node_stopping_criteria.stop(current_node, current_node_labels):
                    if verbose: print(f'>> stopping because {node_stopping_criteria}')

                    stop = True
                    break

            if stop:
                continue

            ## Node expansion
            if verbose: print(f'Expanding node...')

            ### Find best split
            features_indices = list(range(self._features_number))
            best_feature_index, best_parameter, is_continuous = self._find_best_condition(current_node_samples, current_node_labels, features_indices, verbose)

            ### Check gain presence
            if best_feature_index is None:
                if verbose: print(f'>> stop expansion because no best condition was found')

                continue
            
            if verbose: print(f'>> best feature: {best_feature_index} best parameter: {best_parameter} continuous: {is_continuous}')

            ### Convert current node
            current_node.label = None
            current_node.feature_index = best_feature_index
            if is_continuous:
                condition = self._continuous_condition(best_parameter)
            else:
                condition = self._categorical_condition(best_parameter)
            current_node.condition = condition
            self._leaves_count -= 1

            if verbose: print(f'Converting node > condition: {condition}')

            ### Build new children nodes
            if verbose: print(f'Adding children nodes:')

            self._depth = max(self._depth, current_depth+1)
            
            condition_results = np.apply_along_axis(np.vectorize(condition.test), 0, current_node_samples[:, best_feature_index])
            condition_unique_results = np.unique(condition_results)

            if verbose: print(f'parent condition unique results: {condition_unique_results}')

            if len(condition_unique_results) == 1:
                if verbose: print(f'>> stop expansion because it invalidates tree structure')

                current_node.label = self._common_label(current_node_labels)
                current_node.feature_index = None
                current_node.condition = None
                self._leaves_count += 1

                continue

            for unique_result in range(np.max(condition_unique_results)+1):
                partition_indices = current_samples_indices[np.where(condition_results == unique_result)]
                partition_labels = labels[partition_indices]
                
                partition_common_label = self._common_label(partition_labels)
                partition_node = TreePredictorNode(label=partition_common_label)
                current_node.children.append(partition_node)
                self._nodes_count += 1
                self._leaves_count += 1

                if verbose: print(f'- adding child {unique_result} > label: {partition_common_label} samples: {len(partition_labels)} depth: {current_depth+1}')

                nodes_queue.append((partition_node, partition_indices, current_depth+1))

    def _find_best_condition(self, samples:np.ndarray, labels:np.ndarray, available_feature_indices:list[int], verbose:bool = False) -> tuple[int, Any, bool]:
        """Examines all the features associated to the given indices, finding
        the best one and the best paramter (according to the feature type) that
        allow to build the best condition, that is the condition that maximizes
        the decision metric used by this tree.
        The parameters of this condition and the type of the feature are
        returned if a gain condition with respect to the decision metric used
        in the tree is found.

        Args:
            samples (np.ndarray): collection representing the sample to use
            during the evaluation of the features.
            labels (np.ndarray): collection representing the labels that are
            associated to samples.
            available_feature_indices (list[int]): indces of the feature that
            have to be evaluated.
            verbose (bool, optional): determines wheter or not to print
            information during the process. Defaults to False for no prints.

        Returns:
            tuple[int, Any, bool]: best feature index, best condition's parameter
            value and whether or not best feature is continuous if there is a
            gain, (None, None, False) otherwise.
        """

        best_condition_score = 0
        best_feature_index = best_parameter = None
        is_best_feature_continuous = False

        if verbose: print(f'Find best conditions among features {available_feature_indices}:')

        for feature_index in available_feature_indices:
            feature_values = samples[:, feature_index]
            possible_parameters = np.unique(feature_values)
            is_continuous = np.issubdtype(type(feature_values[0]), np.number)

            if is_continuous:
                sorted_values = feature_values.copy()
                sorted_values.sort()
                possible_parameters = []
                for i in range(1, len(sorted_values)):
                    left_value = sorted_values[i-1]
                    right_value = sorted_values[i]
                    if left_value != right_value:
                        mid_point = (left_value + right_value) / 2
                        possible_parameters.append(mid_point)
                    
            if verbose: print(f'- checking feature [{feature_index}] > possible parameters: {len(possible_parameters)} continuous: {is_continuous}')

            for possible_parameter in possible_parameters:
                if is_continuous:
                    condition = self._continuous_condition(possible_parameter)
                else:
                    condition = self._categorical_condition(possible_parameter)
                
                parameter_score = self._decision_metric(feature_values, labels, condition.test)

                if parameter_score > best_condition_score:
                    best_condition_score = parameter_score
                    best_feature_index = feature_index
                    best_parameter = possible_parameter
                    is_best_feature_continuous = is_continuous

        if verbose: print(f'best condition > feature index: {best_feature_index} score: {best_condition_score} parameter: {best_parameter} continuous: {is_best_feature_continuous}')

        return best_feature_index, best_parameter, is_best_feature_continuous

    def _traverse_tree(self, sample:np.ndarray) -> Any:
        """Traverses the tree underlying this tree predictor starting from the
        root node, applying the conditional test of each internal node to the
        provided sample, returning the predicted label for the sample, that is
        the label associated to the final reached leaf node.

        Args:
            sample (np.ndarray): collection representing the sample to test.

        Returns:
            Any: prediction label determined by the tree.
        """

        current_node = self._root

        while not current_node.is_leaf():
            next_node_index = current_node.test(sample)
            current_node = current_node.children[next_node_index]

        final_label = current_node.label

        return final_label

    def _common_label(self, labels:np.ndarray) -> Any:
        """Returns the most common label among the given ones, that is the label
        having the highest count.

        Args:
            labels (np.ndarray): collection of labels to use.

        Returns:
            Any: most common label.
        """

        labels_count = Counter(labels)
        most_common_label = labels_count.most_common()[0][0]

        return most_common_label
    
    def __str__(self) -> str:
        if self._root is None:
            return '<empty tree>'

        result = 'TreePredictor:'
        stack = [(self._root, 0)]
        
        while stack:
            node, depth = stack.pop()
            result += '\n' + '  '*depth + str(node)

            for child_node in node.children[::-1]:
                stack.append((child_node, depth+1))

        return result
