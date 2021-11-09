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

data_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

data = load_analysis_data(paths=paths, level=args.level)
feats_er = data['feats_er']
labels = data['labels_er']

train_ids, test_ids = get_train_test_ids(labels, balanced=True)

train_feats, train_labels = feats_er.loc[train_ids], labels.loc[train_ids]
test_feats, test_labels = feats_er.loc[test_ids], labels.loc[test_ids]

train_dataset = [train_feats, train_labels]
test_dataset = [test_feats, test_labels]

train_pred_labels, test_pred_labels = \
    base_classification(train_dataset, test_dataset, 'dwd', predict=True)

train_misclf_dir = os.path.join(Paths().classification_dir,
                                'misclassified_er',
                                'train')
test_misclf_dir = os.path.join(Paths().classification_dir,
                               'misclassified_er',
                               'test')

get_misclassified_images(train_ids, train_labels, train_pred_labels, 'er',
                         train_misclf_dir)
get_misclassified_images(test_ids, test_labels, test_pred_labels, 'er',
                         test_misclf_dir)
