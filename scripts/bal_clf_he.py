import os
import numpy as np
import pandas as pd

from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_image_feats import load_image_feats
from tma_ajive.classification import base_classification, get_balanced_ids


data = load_image_feats(load_patch_data=False)
feats_he = data['subj_feats_he']

labels_file = os.path.join(Paths().classification_dir, 'subj_labels_er.csv')
labels = pd.read_csv(labels_file, index_col=0)

intersection = list(set(feats_he.index).intersection(set(labels.index)))
feats_he = feats_he.loc[intersection]
labels = labels.loc[intersection]

metrics_list = []
for i in tqdm(range(100)):
    train_id, test_id = get_balanced_ids(labels)

    train_feats, train_labels = feats_he.loc[train_id], labels.loc[train_id]
    test_feats, test_labels = feats_he.loc[test_id], labels.loc[test_id]

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate = \
        base_classification(train_dataset, test_dataset, 'dwd')
    metrics_list.append([acc, tp_rate, tn_rate])

dump(metrics_list, os.path.join(Paths().classification_dir, 'metrics_list_he'))
mean_metrics = np.mean(metrics_list, axis=0)
lower_metrics = np.percentile(metrics_list, 5, axis=0)
upper_metrics = np.percentile(metrics_list, 95, axis=0)
print('Mean accuracy: {}, Mean TP rate: {}, Mean TN rate: {}'\
    .format(mean_metrics[0], mean_metrics[1], mean_metrics[2]))
print('CI of accuracy: ({},{}), CI of TP rate: ({},{}), CI of TN rate:({},{})'\
    .format(lower_metrics[0], upper_metrics[0],
            lower_metrics[1], upper_metrics[1],
            lower_metrics[2], upper_metrics[2]))
