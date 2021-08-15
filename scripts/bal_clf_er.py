import os
import numpy as np
import pandas as pd

from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *


data = load_analysis_data(load_patch_data=False)
feats_er = data['subj_feats_er']
labels = data['subj_labels_er']

metrics_list = []
for i in tqdm(range(10)):
    train_ids, test_ids = get_balanced_ids(labels)

    train_feats, train_labels = feats_er.loc[train_ids], labels.loc[train_ids]
    test_feats, test_labels = feats_er.loc[test_ids], labels.loc[test_ids]

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate = \
        base_classification(train_dataset, test_dataset, 'dwd')
    metrics_list.append([acc, tp_rate, tn_rate])

dump(metrics_list, os.path.join(Paths().classification_dir, 'metrics_list_er'))
print_classification_results(metrics_list)
