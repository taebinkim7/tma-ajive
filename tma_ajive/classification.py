import os
import numpy as np
import pandas as pd
from patch_classifier import DWDClassifier
from sklearn.metrics import confusion_matrix


def base_classification(train_dataset, test_dataset, classifier_type):
    train_feats, train_labels = train_dataset
    test_feats, test_labels = test_dataset

    train_feats = train_feats.to_numpy()
    test_feats = test_feats.to_numpy()
    train_labels = train_labels['er_label'].to_numpy()
    test_labels = test_labels['er_label'].to_numpy()

    # fit classifier
    if classifier_type == 'dwd':
        classifier = DWDClassifier().fit(train_feats, train_labels)

    # calculate evaluation metrics
    acc = classifier.score(test_feats, test_labels)
    predicted_labels = classifier.predict(test_feats)
    tn, fp, fn, tp = confusion_matrix(test_labels, predicted_labels).ravel()
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)

    print('Accuracy: {}, TP rate: {}, TN rate:{}'.format(round(acc, 3),
                                                         round(tp_rate, 3),
                                                         round(tn_rate, 3)))

    return acc, tp_rate, tn_rate

def get_balanced_ids(labels, seed=None):
    # split positive and negative objects
    pos_id = labels[labels['er_label']==1].index
    neg_id = labels[labels['er_label']==0].index
    pos_id = np.random.RandomState(seed=None).permutation(pos_id)
    neg_id = np.random.RandomState(seed=None).permutation(neg_id)

    n = min(len(pos_id), len(neg_id))
    train_id = list(pos_id[:int(.8 * n)]) + list(neg_id[:int(.8 * n)])
    test_id = list(pos_id[int(.8 * n):]) + list(neg_id[int(.8 * n):])

    return train_id, test_id
