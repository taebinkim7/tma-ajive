import os
import numpy as np
import pandas as pd

from patch_classifier import DWDClassifier
from tma_ajive.Paths import Paths
from tma_ajive.load_image_feats import load_image_feats


def base_classification(feats, classifier_type, seed=None):
    labels_file = os.path.join(Paths().classification_dir, 'subj_labels_er.csv')
    labels = pd.read_csv(labels_file, index_col=0)

    intersection = list(set(feats.index).intersection(set(labels.index)))
    feats = feats.loc[intersection]
    labels = labels.loc[intersection]

    feats = feats.to_numpy()
    labels = labels['er_label'].to_numpy().astype(int)

    n = len(labels)
    perm_idx = np.random.RandomState(seed=seed).permutation(np.arange(n))
    train_idx, test_idx = perm_idx[:int(.8 * n)], perm_idx[int(.8 * n):]

    train_feats, test_feats = feats[train_idx], feats[test_idx]
    train_labels, test_labels = labels[train_idx], labels[test_idx]

    if classifier_type == 'dwd':
        classifier = DWDClassifier().fit(train_feats, train_labels)

    train_acc = classifier.score(train_feats, train_labels)
    test_acc = classifier.score(test_feats, test_labels)
    print('The prediction accuracy of the trained {} on the train data is {}.'\
        .format(classifier_type.upper(), train_acc))
    print('The prediction accuracy of the trained {} on the test data is {}.'\
        .format(classifier_type.upper(), test_acc))

    return train_acc, test_acc
