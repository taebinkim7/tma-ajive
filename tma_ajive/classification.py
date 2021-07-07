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

    acc = classifier.score(test_feats, test_labels)
    predicted_labels = classifier.predict(test_feats)
    tn, fp, fn, tp = confusion_matrix(test_labels, predicted_labels).ravel()
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)

    print('Accuracy: {}, TP rate: {}, TN rate:{}'.format(acc, tp_rate, tn_rate))

    return acc, tp_rate, tn_rate
