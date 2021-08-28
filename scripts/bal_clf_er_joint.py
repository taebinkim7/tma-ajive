import os
import numpy as np
import pandas as pd

from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *
from tma_ajive.ajive import fit_ajive


data = load_analysis_data(load_patch_data=False)
feats_he = data['feats_he']
feats_er = data['feats_er']
labels = data['labels_er']

metrics_list = []
for i in tqdm(range(10)):
    train_ids, test_ids = get_train_test_ids(labels, balanced=True)

    train_feats_he, train_feats_er, train_labels = \
        feats_he.loc[train_ids], feats_er.loc[train_ids], labels.loc[train_ids]
    test_feats_er, test_labels = feats_er.loc[test_ids], labels.loc[test_ids]

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
print_classification_results(metrics_list)
