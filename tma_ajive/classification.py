import os
import numpy as np
import pandas as pd
from patch_classifier import DWDClassifier


def base_classification(train_dataset, test_dataset, classifier_type):
    train_feats, train_labels = train_dataset
    test_feats, test_labels = test_dataset

    train_feats = train_feats.to_numpy()
    test_feats = test_feats.to_numpy()
    train_labels = train_labels['er_label'].to_numpy()
    test_labels = test_labels['er_label'].to_numpy()

    if classifier_type == 'dwd':
        classifier = DWDClassifier().fit(train_feats, train_labels)

    train_acc = classifier.score(train_feats, train_labels)
    test_acc = classifier.score(test_feats, test_labels)
    print('The prediction accuracy of the trained {} on the train data is {}.'\
        .format(classifier_type.upper(), train_acc))
    print('The prediction accuracy of the trained {} on the test data is {}.'\
        .format(classifier_type.upper(), test_acc))

    return train_acc, test_acc
