import os
import numpy as np
import pandas as pd

from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import base_classification


data = load_analysis_data(load_patch_data=False)
feats_er = data['subj_feats_er']
labels = data['subj_labels_er']

metrics_list = []
for i in tqdm(range(10)):
    n = len(labels)
    perm_idx = np.random.RandomState(seed=None).permutation(np.arange(n))
    train_idx, test_idx = perm_idx[:int(.8 * n)], perm_idx[int(.8 * n):]

    train_feats, train_labels = feats_er.iloc[train_idx], labels.iloc[train_idx]
    test_feats, test_labels = feats_er.iloc[test_idx], labels.iloc[test_idx]

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate = \
        base_classification(train_dataset, test_dataset, 'dwd')
    metrics_list.append([acc, tp_rate, tn_rate])

dump(metrics_list, os.path.join(Paths().classification_dir, 'metrics_list_er'))
mean_metrics = np.mean(metrics_list, axis=0)
lower_metrics = np.percentile(metrics_list, 5, axis=0)
upper_metrics = np.percentile(metrics_list, 95, axis=0)
print('Mean accuracy: {}, Mean TP rate: {}, Mean TN rate: {}'\
    .format(mean_metrics[0], mean_metrics[1], mean_metrics[2]))
print('CI of accuracy: ({},{}), CI of TP rate: ({},{}), CI of TN rate:({},{})'\
    .format(lower_metrics[0], upper_metrics[0],
            lower_metrics[1], upper_metrics[1],
            lower_metrics[2], upper_metrics[2]))
