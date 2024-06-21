import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from datasets.methods import dataset_information, delete_dataset_features, fill_dataset_samples, extract_samples_labels
from predictors.treepredictors import TreePredictor
from commons.stopping_criteria import NodeImpurityLevel, NodeMinimumSamples
from commons.splitting_criteria import ThresholdCondition, MembershipCondition
from commons.splitting_criteria import entropy, information_gain, gini_impurity_gain, minimum_gain
from commons.losses import zero_one_loss

def main():
    DATASET_FILENAME = 'datasets/mushroom_secondary.csv'
    SAMPLES_NUMBER = 10_000
    NORMALIZATION = True
    DELETION_THRESHOLD = 0.20
    IMPUTATION_VALUE = 'u'

    TEST_SIZE = 0.2
    RANDOM_SEED = 1234
    SHUFFLE = True

    VERBOSE = True

    ## Loading dataset
    mushroom_dataset = pd.read_csv(DATASET_FILENAME, sep=';', nrows=SAMPLES_NUMBER)

    ## Dataset preprocessing
    mushroom_dataset = delete_dataset_features(mushroom_dataset, DELETION_THRESHOLD)
    mushroom_dataset = fill_dataset_samples(mushroom_dataset, IMPUTATION_VALUE)

    information = dataset_information(mushroom_dataset, NORMALIZATION)

    print('Dataset information:')
    for k, v in information.items():
        print(f'{k}: {v}')
    print()

    ## Train-test split
    samples_set, labels_set = extract_samples_labels(mushroom_dataset)
    train_sample_set, test_sample_set, train_labels_set, test_labels_set = train_test_split(samples_set, labels_set, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RANDOM_SEED)

    print('Train set & test set')
    print(len(train_sample_set), len(test_sample_set))
    print(train_sample_set[:5], '\n', test_sample_set[:5], sep='')
    print('Train labels & test labels')
    print(len(train_labels_set), len(test_labels_set))
    print(train_labels_set[:5], '\n', test_labels_set[:5], sep='')
    print()

    ## Tree predictor initialization
    continuous_condition = ThresholdCondition
    categorical_condition = MembershipCondition
    node_stopping_criteria = [NodeMinimumSamples(1)] # [NodeMinimumSamples(15), NodeImpurityLevel(entropy, 0.20)]
    decision_metric = information_gain # minimum_gain, information_gain, gini_impurity_gain

    tree_predictor = TreePredictor(
        continuous_condition=continuous_condition,
        categorical_condition=categorical_condition,
        node_stopping_criteria=node_stopping_criteria,
        decision_metric=decision_metric
    )

    print('Initial tree predictor:')
    print(tree_predictor)
    print()

    ## Training tree
    tree_predictor.fit(train_sample_set, train_labels_set, VERBOSE)

    print('Final tree predictor:')
    print(tree_predictor)
    print(tree_predictor._depth, tree_predictor._nodes_count, tree_predictor._leaves_count)
    print()

    ## Predicting labels
    predictions = tree_predictor.predict(test_sample_set)

    ## Compute loss
    loss = zero_one_loss(test_labels_set, predictions)
    values, count = np.unique(loss, return_counts=True)
    
    print('Loss:')
    print(values, count)
    print()

if __name__ == '__main__':
    main()