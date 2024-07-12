import sys
sys.path.append('..')

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.methods import dataset_information, delete_dataset_features, fill_dataset_samples, extract_samples_labels
from predictors.treepredictors import TreePredictor
from commons.splitting_criteria import ThresholdCondition, MembershipCondition
from commons.splitting_criteria import information_gain, gini_impurity_gain, misclassification_gain, chi_square
from commons.losses import zero_one_loss, samples_error

def main():

    ## Constants
    DATASET_FILENAME = 'datasets/mushroom_secondary.csv'
    SAMPLES_NUMBER = 60_000
    DELETION_THRESHOLD = 0.20
    IMPUTATION_VALUE = 'u'
    NORMALIZATION = True

    RANDOM_SEED = 1234
    TEST_SIZE = 0.2
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

    print(f'Original dataset size: {len(samples_set)}')
    print(f'Train set size:        {len(train_sample_set)}')
    print(f'Test set size:         {len(test_sample_set)}')
    print()
    print(f'Data and label example:')
    print(f'({train_sample_set[0]}, {train_labels_set[0]})')
    print()

    ## ==========================================

    continuous_condition = ThresholdCondition
    categorical_condition = MembershipCondition
    node_stopping_criteria = []
    tree_stopping_criteria = []

    for decision_metric in [misclassification_gain, information_gain, gini_impurity_gain, chi_square]:

        ## Tree predictor initialization
        tree_predictor = TreePredictor(
            continuous_condition=continuous_condition,
            categorical_condition=categorical_condition,
            decision_metric=decision_metric,
            tree_stopping_criteria=tree_stopping_criteria,
            node_stopping_criteria=node_stopping_criteria
        )

        print(f'Initial {decision_metric.__name__} tree predictor:')
        print(tree_predictor)
        print(f'Tree depth:  {tree_predictor._depth}')
        print(f'Total nodes: {tree_predictor._nodes_count}')
        print(f'Leaf nodes:  {tree_predictor._leaves_count}')
        print()

        ## Training tree predictor
        tree_predictor.fit(train_sample_set, train_labels_set, VERBOSE)

        print(f'Trained {decision_metric.__name__} tree predictor:')
        print(tree_predictor)
        print(f'Tree depth:  {tree_predictor._depth}')
        print(f'Total nodes: {tree_predictor._nodes_count}')
        print(f'Leaf nodes:  {tree_predictor._leaves_count}')

        ## Train and test errors
        predictor = tree_predictor.predict

        train_error = samples_error(predictor, zero_one_loss, train_labels_set, train_sample_set)

        test_error = samples_error(predictor, zero_one_loss, test_labels_set, test_sample_set)

        print(f'Train error: {train_error}')
        print(f'Test error:  {test_error}')

if __name__ == '__main__':
    main()