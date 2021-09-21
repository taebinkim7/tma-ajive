import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from joblib import load
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tma_ajive.Paths import Paths
from tma_ajive.cnn_classification import cnn_classification


tensors = load(os.path.join(Paths().features_dir, 'core_tensors_er'))
labels = pd.read_csv(os.path.join(Paths().classification_dir,
                                  'core_labels_er.csv'), index_col=0)

intersection = list(set(tensors.keys()).intersection(set(labels.index)))
X = np.array([tensors[id] for id in intersection])
y = labels.loc[intersection]['er_label'].to_numpy().astype(int)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = .001

metrics_list = []
for i in tqdm(range(10)):
    acc, tp_rate, tn_rate, precision = \
        cnn_classification(epochs=EPOCHS, batch_size=BATCH_SIZE,
                           learning_rate=LEARNING_RATE)
    metrics_list.append([acc, tp_rate, tn_rate, precision])

print_classification_results(metrics_list)
