import os
import numpy as np
import pandas as pd

from joblib import dump
from tma_ajive.load_analysis_data import load_analysis_data
from patch_classifier import DWDClassifier
from tma_ajive.Paths import Paths

data = load_analysis_data(load_patch_data=False)
# save dataset
dump(data, os.path.join(Paths().classification_dir, 'data'))

feats_er = data['feats_er']
labels = data['labels_er']

ids = labels.index
feats = feats_er.to_numpy()
labels = labels['er_label'].to_numpy()

# train DWD classifier using all data
classifier = DWDClassifier().fit(feats, labels)

# save DWD classifier
classifier.save(os.path.join(Paths().classification_dir, 'dwd_all'))
