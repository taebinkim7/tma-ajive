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
for i in tqdm(range(10)):
    train_id, test_id = get_balanced_ids(labels)

    train_feats_he, train_feats_er, train_labels = \
        feats_he.loc[train_id], feats_er.loc[train_id], labels.loc[train_id]
    test_feats_er, test_labels = feats_er.loc[test_id], labels.loc[test_id]

    ajive = fit_ajive(train_feats_he, train_feats_er, train_labels)
    er_joint_loadings = ajive.blocks['er'].joint.loadings_
    train_feats_er_joint = train_feats_er @ er_joint_loadings
    test_feats_er_joint = test_feats_er @ er_joint_loadings
    train_feats = train_feats_er_joint
    test_feats = test_feats_er_joint

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate = \
        base_classification(train_dataset, test_dataset, 'dwd')
    metrics_list.append([acc, tp_rate, tn_rate])

dump(metrics_list, os.path.join(Paths().classification_dir,
     'metrics_list_er_joint'))
mean_metrics = np.mean(metrics_list, axis=0)
lower_metrics = np.percentile(metrics_list, 5, axis=0)
upper_metrics = np.percentile(metrics_list, 95, axis=0)
print('Mean accuracy: {}, Mean TP rate: {}, Mean TN rate: {}'\
    .format(round(mean_metrics[0], 3),
            round(mean_metrics[1], 3),
            round(mean_metrics[2], 3)))
print('CI of accuracy: ({},{}), CI of TP rate: ({},{}), CI of TN rate:({},{})'\
    .format(round(lower_metrics[0], 3), round(upper_metrics[0], 3),
            round(lower_metrics[1], 3), round(upper_metrics[1], 3),
            round(lower_metrics[2], 3), round(upper_metrics[2], 3)))
