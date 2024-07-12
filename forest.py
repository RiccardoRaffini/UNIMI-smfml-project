import sys
sys.path.append('..')

import pandas as pd
from sklearn.model_selection import train_test_split

from datasets.methods import dataset_information, delete_dataset_features, fill_dataset_samples, extract_samples_labels
from predictors.treepredictors import TreePredictor
from commons.splitting_criteria import ThresholdCondition, MembershipCondition
from commons.splitting_criteria import AllFeaturesSelector, RandomFeaturesSelector
from commons.splitting_criteria import information_gain, gini_impurity_gain, misclassification_gain, chi_square
from commons.losses import zero_one_loss, samples_error
from commons.plotting import plot_train_test_error, plot_confusion_matrices

def main():
    
    ## Constants

    ## Dataset loading and preprocessing

    ## Tree predictors parameters

    features_selector = AllFeaturesSelector()

    selected = features_selector.select(12)

    print(features_selector)
    print(selected)

    features_selector = RandomFeaturesSelector(5, 1234)

    print(features_selector)

    selected = features_selector.select(12)
    print(selected)

    selected = features_selector.select(12)
    print(selected)

    selected = features_selector.select(12)
    print(selected)

    ## Forest initialization

    ## Forest training

    ## Forest testing


if __name__ == '__main__':
    main()