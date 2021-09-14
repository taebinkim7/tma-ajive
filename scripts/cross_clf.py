import os
import numpy as np

from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *


ref_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9741/'
target_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9830_normed/'
ref_data = load_analysis_data(paths=Paths(ref_dir))
target_data = load_analysis_data(paths=Paths(target_dir))

train_feats, train_labels = ref_data['feats_er'], ref_data['labels_er']
test_feats, test_labels = target_data['feats_er'], target_data['labels_er']

train_dataset = [train_feats, train_labels]
test_dataset = [test_feats, test_labels]

base_classification(train_dataset, test_dataset, 'wdwd')
