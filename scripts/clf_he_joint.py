import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *
from tma_ajive.ajive import fit_ajive


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
parser.add_argument('--iter', type=int, default=10)
args = parser.parse_args()

data_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

data = load_analysis_data(paths=paths, level=args.level)
feats_he = data['feats_he']
feats_er = data['feats_er']
labels = data['labels_er']

metrics_list = []
for i in tqdm(range(10)):
    train_ids, test_ids = get_train_test_ids(labels)

    train_feats_he, train_feats_er, train_labels = \
        feats_he.loc[train_ids], feats_er.loc[train_ids], \
        labels.loc[train_ids]
    test_feats_he, test_labels = feats_he.loc[test_ids], labels.loc[test_ids]

    ajive = fit_ajive(train_feats_he, train_feats_er, train_labels)
    he_joint_loadings = ajive.blocks['he'].joint.loadings_
    train_feats_he_joint = train_feats_he @ he_joint_loadings
    test_feats_he_joint = test_feats_he @ he_joint_loadings
    train_feats = train_feats_he_joint
    test_feats = test_feats_he_joint

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate, precision = \
        base_classification(train_dataset, test_dataset, 'wdwd')
    metrics_list.append([acc, tp_rate, tn_rate, precision])

dump(metrics_list, os.path.join(paths.classification_dir,
     'metrics_list_he_joint'))
print_classification_results(metrics_list)
