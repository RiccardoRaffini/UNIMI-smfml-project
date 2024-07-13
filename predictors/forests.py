import numpy as np
import math
from typing import Any, Callable, Type, Tuple
from collections import Counter

from commons.splitting_criteria import Condition, RandomFeaturesSelector
from commons.stopping_criteria import TreeStopCondition, NodeStopCondition
from predictors.treepredictors import TreePredictor

class TreePredictorRandomForest:
    """Instances of this class represent a collection of generic tree predictors,
    trainable to learn an ensemble of decision trees based on decision with a
    generic number of outcomes and appliable to any kind of features.
    Different aspects of the trees computation and training behavior can be
    customized to determine the kind of the obtained trees, their charcteristics
    and their number.
    """

    def __init__(self,
        tree_predictors_number:int,
        continuous_condition:Type[Condition], categorical_condition:Type[Condition],
        decision_metric:Callable[[np.ndarray, np.ndarray, Callable[[Any], int]], np.number] = None,
        tree_stopping_criteria:list[TreeStopCondition] = None,
        node_stopping_criteria:list[NodeStopCondition] = None,
        random_seed:int = None
        ) -> None:
        """Initialize a new TreePredictorRandomForest instance by specifying the
        number of tree predictor to use and the criteria required for their
        training and computational behavior.

        Args:
            tree_predictors_number (int): number of tree predictors that should
            compose the random forest.
            continuous_condition (Type[Condition]): continuous condition used
            by contained trees to handle continuous features.
            categorical_condition (Type[Condition]): categorical condition used
            by contained trees to handle catehorical features.
            decision_metric (Callable[[np.ndarray, np.ndarray, Callable[[Any], int]], np.number], optional):
            decision metric usaed by contained trees for evaluating possible
            decision of conditional statements. Defaults to None for using a
            base implementation of the information gain metric based on the entropy.
            tree_stopping_criteria (list[TreeStopCondition], optional): collection
            of stopping criteria used by contained trees to check during training.
            Defaults to None for using only a default criteria limiting the 
            epth of the tree to 100.
            node_stopping_criteria (list[NodeStopCondition], optional): collection
            of stopping criteria for nodes to check during training used by
            contained trees. Defaults to None for not using any criteria.
        """
        
        ## Forest structure
        self._tree_predictors_number = tree_predictors_number
        self._tree_predictors:list[TreePredictor] = []

        ## Computation criteria
        self._features_number:int = None
        self._features_selection_number:int = None
        self._random_seed = random_seed

        ## Tree predictors parameters
        self._tree_continuous_condition = continuous_condition
        self._tree_categorical_condition = categorical_condition
        self._tree_decision_metric = decision_metric
        self._tree_tree_stopping_criteria = tree_stopping_criteria
        self._tree_node_stopping_criteria = node_stopping_criteria

    def fit(self, samples:np.ndarray, labels:np.ndarray, verbose:bool = False) -> None:
        """Trains the underlying model of this tree predictor random forest
        using the given samples and their associated labels.

        Args:
            samples (np.ndarray): collection of samples usable to train the model.
            labels (np.ndarray): collection of labels associated to the samples.
            verbose (bool, optional): determines wheter or not to print
            information during the process. Defaults to False for no prints.
        """
        
        ## Information about samples
        _, samples_dimensionality = samples.shape
        self._features_number = samples_dimensionality
        self._features_selection_number = math.ceil(math.sqrt(self._features_number))

        ## New forest initialization
        self._tree_predictors = []
        if self._random_seed:
            np.random.seed(self._random_seed)

        if verbose: print(f'Initialized new random forest')

        ## Trees random seeds
        tree_bagging_seeds = np.random.randint(1, 2**16, self._tree_predictors_number)
        tree_selector_seeds = np.random.randint(1, 2**16, self._tree_predictors_number)

        if verbose: print(f'Trees bagging seeds: {tree_bagging_seeds}\nTrees selector seeds: {tree_selector_seeds}')

        ## Individual tree training
        if verbose: print(f'Training trees...')

        for i in range(self._tree_predictors_number):
            if verbose: print(f'Training tree predictor [{i}]...')

            ## Samples boosting aggregation
            if verbose: print(f'bagging samples...')
            samples_bag, labels_bag = self._samples_bagging(samples, labels, tree_bagging_seeds[i])

            ## Features selector
            if verbose: print(f'initializing selector...')

            features_selector = RandomFeaturesSelector(self._features_selection_number, tree_selector_seeds[i])

            ## Tree predictor initialization
            if verbose: print(f'initializing tree predictor')

            tree_predictor = TreePredictor(
                self._tree_continuous_condition,
                self._tree_categorical_condition,
                self._tree_decision_metric,
                self._tree_tree_stopping_criteria,
                self._tree_node_stopping_criteria,
                features_selector
            )

            ## Training tree predictor
            if verbose: print(f'training tree predictor...')

            tree_predictor.fit(samples_bag, labels_bag, verbose)

            ## Adding tree to forest
            self._tree_predictors.append(tree_predictor)

            if verbose: print(f'tree predictor added to forest')

    def predict(self, samples:np.ndarray) -> np.ndarray:
        """Predicts and retruns the labels of the given samples obtained using
        the tree predictors underlying this random forest.

        Args:
            samples (np.ndarray): collection of samples to which predict labels.

        Returns:
            np.ndarray: prediction labels determined by the model. 
        """

        samples_predictions = np.array([self._predict(sample) for sample in samples])

        return samples_predictions

    def _predict(self, sample:np.ndarray) -> Any:
        """Predicts and retruns the labels of the given sample obtained as the
        majority vote of the predictions performed by the tree predictors
        composing this random forest.

        Args:
            sample (np.ndarray): individual sample to which predict label.

        Returns:
            Any: prediction label determined by the tree
        """
        
        trees_predictions = np.array([tree_predictor.predict([sample]) for tree_predictor in self._tree_predictors])
        trees_predictions = trees_predictions.flatten()
        forest_prediction = self._common_label(trees_predictions)

        return forest_prediction

    def _samples_bagging(self, samples:np.ndarray, labels:np.ndarray, random_seed:int = None) -> Tuple[np.ndarray, np.ndarray]:
        """Performs bagging procedure on the given collection of samples and
        labels, returning a new collection with the same size, but containing
        randomly extracted samples with replacements.

        Args:
            samples (np.ndarray): original collection of samples.
            labels (np.ndarray): original collection of labels associated to samples.
            random_seed (int, optional): random seed to use in samples extraction.
            Defaults to None for a completely random choice.

        Returns:
            Tuple[np.ndarray, np.ndarray]: pair containing the new sample and
            labels random bags.
        """
        
        samples_number, _ = samples.shape

        if random_seed:
            np.random.seed(random_seed)
        
        bag_indices = np.random.choice(samples_number, samples_number, replace=True)
        samples_bag = samples[bag_indices]
        labels_bag = labels[bag_indices]

        return samples_bag, labels_bag

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
        if self._tree_predictors is None:
            return '<empty random forest>'
        
        result = f'Random forest composed of {self._tree_predictors_number} tree predictors:'
        for i, tree_predictor in enumerate(self._tree_predictors):
            result += f'\nTree predictor [{i}]:'
            result += f'\n{str(tree_predictor)}'

        return result