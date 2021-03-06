import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
parser.add_argument('--iter', type=int, default=10)
parser.add_argument('--target', type=str, default='labels_er')
args = parser.parse_args()

data_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

if 'surv' in args.target: # e.g., surv_3
    data = load_analysis_data(paths=paths, level=args.level,
                              types=['feats_er', 'survival'])
    feats_er = data['feats_er']
    k = int(args.target.split('_')[1]) # k years survival
    labels = data['survival']
    labels = labels.loc[(labels['surv_mos'] > 12 * k) | (labels['death'] == 1)]
    labels[args.target] = [int(mo > 12 * k) for mo in labels['surv_mos']]
    labels = labels.drop(columns=['surv_mos', 'death'])
else:
    data = load_analysis_data(paths=paths, level=args.level,
                              types=['feats_er', args.target])
    feats_er = data['feats_er']
    labels = data[args.target]

n_pos = np.sum(labels.to_numpy()==1)
n_neg = np.sum(labels.to_numpy()==0)
print('No. positive objects: {}, No. negative objects: {}'.format(n_pos, n_neg))

metrics_list = []
for i in tqdm(range(args.iter)):
    train_ids, test_ids = get_train_test_ids(labels)

    train_feats, train_labels = feats_er.loc[train_ids], labels.loc[train_ids]
    test_feats, test_labels = feats_er.loc[test_ids], labels.loc[test_ids]

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate, precision, f1_score, auc = \
        base_classification(train_dataset, test_dataset, 'wdwd')
    metrics_list.append([acc, tp_rate, tn_rate, precision, f1_score, auc])

# dump(metrics_list, os.path.join(paths.classification_dir, 'metrics_list_er'))
print_classification_results(metrics_list)
