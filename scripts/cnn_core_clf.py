import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from argparse import ArgumentParser
from joblib import load
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tma_ajive.Paths import Paths
from tma_ajive.classification import print_classification_results
from tma_ajive.nn_classification import nn_classification


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
parser.add_argument('--iter', type=int, default=10)
args = parser.parse_args()

data_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

data = load_analysis_data(paths=paths, level=args.level)
tensors = load(os.path.join(paths.features_dir, args.level + '_tensors_er'))
labels = pd.read_csv(os.path.join(paths.classification_dir,
                                  args.level + '_labels_er.csv'), index_col=0)

intersection = list(set(tensors.keys()).intersection(set(labels.index)))
X = np.array([tensors[id] for id in intersection])
y = labels.loc[intersection].to_numpy().reshape(-1).astype(int)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = .001

metrics_list = []
for i in tqdm(range(10)):
    acc, tp_rate, tn_rate, precision = \
        nn_classification(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
                           learning_rate=LEARNING_RATE)
    metrics_list.append([acc, tp_rate, tn_rate, precision])

print_classification_results(metrics_list)
