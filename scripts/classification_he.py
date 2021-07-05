import os
import numpy as np
import pandas as pd

from joblib import dump
from tqdm import tqdm
from patch_classifier import DWDClassifier
from tma_ajive.Paths import Paths
from tma_ajive.load_image_feats import load_image_feats

# TODO: Add argparse for classifier type (e.g., 'dwd') and level (e.g., 'core', 'subj')

def classification_he(classifier_type, seed=None):
    data = load_image_feats(load_patch_feats=False)
    subj_feats_he = data['subj_feats_he']
    subj_feats_he_only = data['subj_feats_he_only']
    feats = subj_feats_he.append(subj_feats_he_only)

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

test_acc_list = []
for i in tqdm(range(100)):
    _, test_acc = classification('er', 'subj', 'dwd')
    test_acc_list.append(test_acc)

# dump(test_acc_list, os.path.join(Paths().classification_dir, 'test_acc_list'))
mean_test_acc = np.mean(test_acc_list)
print(mean_test_acc)
