import os
import numpy as np
import pandas as pd

from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import base_classification, get_balanced_ids
from tma_ajive.ajive import fit_ajive


data = load_analysis_data(load_patch_data=False)
feats_he = data['subj_feats_he']
feats_er = data['subj_feats_er']
labels = data['subj_labels_er']

metrics_list = []
for i in tqdm(range(100)):
    train_id, test_id = get_balanced_ids(labels)

    train_feats_he, train_feats_er, train_labels = \
        feats_he.loc[train_id], feats_er.loc[train_id], labels.loc[train_id]
    test_feats_he, test_labels = feats_he.loc[test_id], labels.loc[test_id]

    ajive = fit_ajive(train_feats_he, train_feats_er, train_labels)
    he_joint_loadings = ajive.blocks['he'].joint.loadings_
    train_feats_he_joint = train_feats_he @ he_joint_loadings
    test_feats_he_joint = test_feats_he @ he_joint_loadings
    train_feats = pd.concat([train_feats_he, train_feats_he_joint], axis=1)
    test_feats = pd.concat([test_feats_he, test_feats_he_joint], axis=1)

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate = \
        base_classification(train_dataset, test_dataset, 'dwd')
    metrics_list.append([acc, tp_rate, tn_rate])

dump(metrics_list, os.path.join(Paths().classification_dir,
     'metrics_list_he_he_joint'))
mean_metrics = np.mean(metrics_list, axis=0)
lower_metrics = np.percentile(metrics_list, 5, axis=0)
upper_metrics = np.percentile(metrics_list, 95, axis=0)
print('Mean accuracy: {}, Mean TP rate: {}, Mean TN rate: {}'\
    .format(mean_metrics[0], mean_metrics[1], mean_metrics[2]))
print('CI of accuracy: ({},{}), CI of TP rate: ({},{}), CI of TN rate:({},{})'\
    .format(lower_metrics[0], upper_metrics[0],
            lower_metrics[1], upper_metrics[1],
            lower_metrics[2], upper_metrics[2]))
