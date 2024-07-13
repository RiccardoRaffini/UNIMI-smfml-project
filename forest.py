import sys
sys.path.append('..')

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from datasets.methods import dataset_information, delete_dataset_features, fill_dataset_samples, extract_samples_labels
from predictors.treepredictors import TreePredictor
from predictors.forests import TreePredictorRandomForest
from commons.splitting_criteria import ThresholdCondition, MembershipCondition, RandomFeaturesSelector
from commons.splitting_criteria import information_gain
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

    TREES_NUMBER = 5
    FEATURES_NUMBER = 8
    VERBOSE = True

    ## Dataset loading and preprocessing
    mushroom_dataset = pd.read_csv(DATASET_FILENAME, sep=';', nrows=SAMPLES_NUMBER)
    mushroom_dataset = delete_dataset_features(mushroom_dataset, DELETION_THRESHOLD)
    mushroom_dataset = fill_dataset_samples(mushroom_dataset, IMPUTATION_VALUE)
    information = dataset_information(mushroom_dataset, NORMALIZATION)

    print('Dataset information:')
    for k, v in information.items():
        print(f'{k}: {v}')
    print()

    ## Train and test split
    samples_set, labels_set = extract_samples_labels(mushroom_dataset)
    train_sample_set, test_sample_set, train_labels_set, test_labels_set = train_test_split(samples_set, labels_set, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RANDOM_SEED)

    print(f'Original dataset size: {len(samples_set)}')
    print(f'Train set size:        {len(train_sample_set)}')
    print(f'Test set size:         {len(test_sample_set)}')
    print()
    print(f'Data and label example:')
    print(f'({train_sample_set[0]}, {train_labels_set[0]})')
    print()

    ## Hyperparameters (tree)
    continuous_condition = ThresholdCondition
    categorical_condition = MembershipCondition
    node_stopping_criteria = []
    tree_stopping_criteria = []
    decision_metric = information_gain

    ## Forest definition
    random_forest = TreePredictorRandomForest(
        tree_predictors_number=TREES_NUMBER,
        continuous_condition=continuous_condition,
        categorical_condition=categorical_condition,
        decision_metric=decision_metric,
        tree_stopping_criteria=tree_stopping_criteria,
        node_stopping_criteria=node_stopping_criteria,
        random_seed=RANDOM_SEED
    )

    print(f'Initial random forest predictor:')
    print(random_forest)
    print(f'Trees depths:  {[tree._depth for tree in random_forest._tree_predictors]}')
    print(f'Trees total nodes: {[tree._nodes_count for tree in random_forest._tree_predictors]}')
    print(f'Trees leaf nodes:  {[tree._leaves_count for tree in random_forest._tree_predictors]}')
    print()

    ## Forest training
    random_forest.fit(train_sample_set, train_labels_set, VERBOSE)

    print(f'Trained random forest predictor:')
    print(random_forest)
    print(f'Trees depths:  {[tree._depth for tree in random_forest._tree_predictors]}')
    print(f'Trees total nodes: {[tree._nodes_count for tree in random_forest._tree_predictors]}')
    print(f'Trees leaf nodes:  {[tree._leaves_count for tree in random_forest._tree_predictors]}')
    print()

    ## Train and test errors
    predictor = random_forest.predict

    train_error = samples_error(predictor, zero_one_loss, train_labels_set, train_sample_set)

    test_error = samples_error(predictor, zero_one_loss, test_labels_set, test_sample_set)

    print(f'Train error: {train_error}')
    print(f'Test error:  {test_error}')
    print()


if __name__ == '__main__':
    main()