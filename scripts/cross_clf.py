import os
import numpy as np

from argparse import ArgumentParser
from tma_ajive.Paths import Paths
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import *


parser = ArgumentParser()
parser.add_argument('--train_dir', type=str, required=True)
parser.add_argument('--test_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
args = parser.parse_args()

train_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.train_dir)
test_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.test_dir)
train_data = load_analysis_data(paths=Paths(train_dir), level=args.level)
test_data = load_analysis_data(paths=Paths(test_dir), level=args.level)

train_feats, train_labels = train_data['feats_er'], train_data['labels_er']
test_feats, test_labels = test_data['feats_er'], test_data['labels_er']

train_dataset = [train_feats, train_labels]
test_dataset = [test_feats, test_labels]

base_classification(train_dataset, test_dataset, 'wdwd')
