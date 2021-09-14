import os
import numpy as np
import pandas as pd

from glob import glob
from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *


ref_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9741/'
target_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9830_normed/'
ref_data = load_analysis_data(paths=Paths(ref_dir))
target_data = load_analysis_data(paths=Paths(target_dir))
feats_er = data['feats_er']
labels = data['labels_er']

metrics_list = []
for i in tqdm(range(10)):
    train_ids, test_ids = get_train_test_ids(labels)

    train_feats, train_labels = ref_data['feats_er'], ref_data['labels_er']
    test_feats, test_labels = target_data['feats_er'], target_data['labels_er']

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    acc, tp_rate, tn_rate = \
        base_classification(train_dataset, test_dataset, 'wdwd')
    metrics_list.append([acc, tp_rate, tn_rate])

# dump(metrics_list, os.path.join(Paths().classification_dir, 'metrics_list_er'))
print_classification_results(metrics_list)
