import sys
sys.path.append('..')

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split

from datasets.methods import delete_dataset_features, fill_dataset_samples, extract_samples_labels
from predictors.treepredictors import TreePredictor
from commons.stopping_criteria import TreeMaximumDepth, TreeMaximumLeaves, TreeMaximumNodes, NodeMinimumSamples, NodeImpurityLevel
from commons.splitting_criteria import ThresholdCondition, MembershipCondition
from commons.splitting_criteria import information_gain, entropy
from commons.losses import zero_one_loss, samples_error, create_folds, cross_validation, holdout_cross_validation
from commons.plotting import format_hyperparameters

def main():

    ## DATASET PREPROCESSING
    
    ## Constants
    DATASET_FILENAME = 'datasets/mushroom_secondary.csv'
    SAMPLES_NUMBER = 10_000
    DELETION_THRESHOLD = 0.20
    IMPUTATION_VALUE = 'u'
    NORMALIZATION = True
    RANDOM_SEED = 1234

    ## Loading dataset
    mushroom_dataset = pd.read_csv(DATASET_FILENAME, sep=';')
    mushroom_dataset = mushroom_dataset.sample(SAMPLES_NUMBER, random_state=RANDOM_SEED)

    ## Dataset preprocessing
    mushroom_dataset = delete_dataset_features(mushroom_dataset, DELETION_THRESHOLD)
    mushroom_dataset = fill_dataset_samples(mushroom_dataset, IMPUTATION_VALUE)
    samples_set, labels_set = extract_samples_labels(mushroom_dataset)

    ## TRAIN AND TEST SPLIT

    print(f'============ TRAIN AND TEST SPLIT ============')

    ## Constants
    RANDOM_SEED = 1234
    TEST_SIZE = 0.2
    SHUFFLE = True

    ## Train and test split

    train_sample_set, test_sample_set, train_labels_set, test_labels_set = train_test_split(samples_set, labels_set, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RANDOM_SEED)

    print(f'Original dataset size: {len(samples_set)}')
    print(f'Train set size:        {len(train_sample_set)}')
    print(f'Test set size:         {len(test_sample_set)}')
    print()

    ## Hyperparameters (no limitations)
    continuous_condition = ThresholdCondition
    categorical_condition = MembershipCondition
    node_stopping_criteria = []
    tree_stopping_criteria = []
    decision_metric = information_gain

    ## Model definition
    tree_predictor = TreePredictor(
        continuous_condition=continuous_condition,
        categorical_condition=categorical_condition,
        decision_metric=decision_metric,
        tree_stopping_criteria=tree_stopping_criteria,
        node_stopping_criteria=node_stopping_criteria
    )

    ## Model training
    tree_predictor.fit(train_sample_set, train_labels_set)

    print(f'Tree depth: {tree_predictor._depth}')
    print(f'Tree total nodes: {tree_predictor._nodes_count}')
    print(f'Tree leaf nodes: {tree_predictor._leaves_count}')
    print()

    ## Model evaluation
    predictor = tree_predictor.predict
    train_error = samples_error(predictor, zero_one_loss, train_labels_set, train_sample_set)
    test_error = samples_error(predictor, zero_one_loss, test_labels_set, test_sample_set)

    print(f'Train error: {train_error}')
    print(f'Test error:  {test_error}')
    print()

    ## CROSS VALIDATION

    print(f'============ CROSS VALIDATION ============')

    ## Constants
    FOLDS = 5
    RANDOM_SEED = 1234
    VERBOSE = True

    ## Folds split
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(samples_set)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(labels_set)

    data_folds = create_folds(samples_set, FOLDS)
    labels_folds = create_folds(labels_set, FOLDS)

    print(f'Original dataset size: {len(samples_set)}')
    print(f'Folds number:          {len(data_folds)}')
    print(f'folds size:            {len(data_folds[0])}')
    print()

    ## Hyperparameters (no limitations)
    continuous_condition = ThresholdCondition
    categorical_condition = MembershipCondition
    node_stopping_criteria = []
    tree_stopping_criteria = []
    decision_metric = information_gain

    ## Model definition
    tree_predictor = TreePredictor(
        continuous_condition=continuous_condition,
        categorical_condition=categorical_condition,
        decision_metric=decision_metric,
        tree_stopping_criteria=tree_stopping_criteria,
        node_stopping_criteria=node_stopping_criteria
    )

    ## Cross validation
    cross_validation_value = cross_validation(tree_predictor, zero_one_loss, data_folds, labels_folds, VERBOSE)

    print(f'final cross validation estimation: {cross_validation_value}')
    print()

    ## HOLDOUT CROSS VALIDATION

    print(f'============ HOLDOUT CROSS VALIDATION ============')

    ## Constants
    SPLITS = (0.75, 0.15, 0.10)
    SHUFFLE = True
    RANDOM_SEED = 1234
    VERBOSE = True

    ## Hyperparameters ranges
    TREE_MAXIMUM_DEPTH_START, TREE_MAXIMUM_DEPTH_END, TREE_MAXIMUM_DEPTH_STEP = 15, 26, 10
    TREE_MAXIMUM_LEAVES_START, TREE_MAXIMUM_LEAVES_END, TREE_MAXIMUM_LEAVES_STEP = 50, 251, 50
    TREE_MAXIMUM_NODES_START, TREE_MAXIMUM_NODES_END, TREE_MAXIMUM_NODES_STEP = 150, 451, 100
    NODE_MINIMUM_SAMPLES_START, NODE_MINIMUM_SAMPLES_END, NODE_MINIMUM_SAMPLES_STEP = 0, 151, 25
    IMPURITY_LEVEL_START, IMPURITY_LEVEL_END, IMPURITY_LEVEL_STEP = 0.0, 0.26, 0.25

    ## Hyperparameters Grid
    continuous_condition_range = [ThresholdCondition]
    categorical_condition_range = [MembershipCondition]
    decision_metric_range = [information_gain]
    maximum_depths = [TreeMaximumDepth(depth) for depth in np.arange(TREE_MAXIMUM_DEPTH_START, TREE_MAXIMUM_DEPTH_END, TREE_MAXIMUM_DEPTH_STEP)]
    maximum_leaves = [TreeMaximumLeaves(number) for number in np.arange(TREE_MAXIMUM_LEAVES_START, TREE_MAXIMUM_LEAVES_END, TREE_MAXIMUM_LEAVES_STEP)]
    maximum_nodes = [TreeMaximumNodes(number) for number in np.arange(TREE_MAXIMUM_NODES_START, TREE_MAXIMUM_NODES_END, TREE_MAXIMUM_NODES_STEP)]
    tree_stopping_criteria_range = np.array(np.meshgrid(maximum_depths, maximum_leaves, maximum_nodes)).T.reshape(-1, 3)
    minimum_samples = [NodeMinimumSamples(number) for number in np.arange(NODE_MINIMUM_SAMPLES_START, NODE_MINIMUM_SAMPLES_END, NODE_MINIMUM_SAMPLES_STEP)]
    impurity_levels = [NodeImpurityLevel(entropy, threshold) for threshold in np.arange(IMPURITY_LEVEL_START, IMPURITY_LEVEL_END, IMPURITY_LEVEL_STEP)]
    node_stopping_criteria_range = np.array(np.meshgrid(minimum_samples, impurity_levels)).T.reshape(-1, 2)

    hyperparameters_grid = list(itertools.product(
        continuous_condition_range,
        categorical_condition_range,
        decision_metric_range,
        tree_stopping_criteria_range,
        node_stopping_criteria_range
    ))

    print(f'Grid dimensions: {len(hyperparameters_grid)}')
    print(f'Hyperparameters example: {format_hyperparameters(hyperparameters_grid[0])}')
    print()

    ## Holdout Cross Validation
    hcv_value, best_hyperparameters = holdout_cross_validation(TreePredictor, zero_one_loss, samples_set, labels_set, SPLITS, hyperparameters_grid, SHUFFLE, RANDOM_SEED, VERBOSE)

    ## Best hyperparametrs
    best_hyperparameters_configuration = format_hyperparameters(best_hyperparameters)

    print(f'final nested cross validation value: {hcv_value}')
    print(f'Best hyperparameters:')
    print(f'continuous condition: {best_hyperparameters_configuration[0]}')
    print(f'categorical condition: {best_hyperparameters_configuration[1]}')
    print(f'decision metric: {best_hyperparameters_configuration[2]}')
    print(f'tree stopping criteria: {best_hyperparameters_configuration[3]}')
    print(f'node stopping criteria: {best_hyperparameters_configuration[4]}')
    print()

    ## Best model training
    continuous_condition = best_hyperparameters[0]
    categorical_condition = best_hyperparameters[1]
    decision_metric = best_hyperparameters[2]
    tree_stopping_criteria = best_hyperparameters[3]
    node_stopping_criteria = best_hyperparameters[4]

    best_model = TreePredictor(continuous_condition, categorical_condition, decision_metric, tree_stopping_criteria, node_stopping_criteria)

    train_sample_set, test_sample_set, train_labels_set, test_labels_set = train_test_split(samples_set, labels_set, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RANDOM_SEED)

    best_model.fit(train_sample_set, train_labels_set)

    ## Best model evaluation
    predictor = best_model.predict
    train_error = samples_error(predictor, zero_one_loss, train_labels_set, train_sample_set)
    test_error = samples_error(predictor, zero_one_loss, test_labels_set, test_sample_set)

    print(f'Train error: {train_error}')
    print(f'Test error:  {test_error}')
    print()

if __name__ == '__main__':
    main()