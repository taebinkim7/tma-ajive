import os
import numpy as np
import pandas as pd

from argparse import ArgumentParser
from glob import glob
from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
parser.add_argument('--iter', type=int, default=10)
args = parser.parse_args()

data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

data = load_analysis_data(paths=paths, level=args.level)
feats_er = data['feats_er']
labels = data['labels_er']

metrics_list = []
for i in tqdm(range(10)):
    train_ids, test_ids = get_train_test_ids(labels)

    train_feats, train_labels = feats_er.loc[train_ids], labels.loc[train_ids]
    test_feats, test_labels = feats_er.loc[test_ids], labels.loc[test_ids]

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate, precision = \
        base_classification(train_dataset, test_dataset, 'dwd')
    metrics_list.append([acc, tp_rate, tn_rate, precision])

# dump(metrics_list, os.path.join(Paths().classification_dir, 'metrics_list_er'))
print_classification_results(metrics_list)
