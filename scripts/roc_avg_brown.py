import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from argparse import ArgumentParser
from joblib import dump
from patch_classifier import WDWDClassifier
from tma_ajive.Paths import Paths
from tma_ajive.classification import get_roc
from tma_ajive.load_analysis_data import load_analysis_data


parser = ArgumentParser(description='Data directory')
parser.add_argument('--data_dir', type=str, action='store')
args = parser.parse_args()

data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)
# data = load_analysis_data(paths=paths)
clf_dir = paths.classification_dir

# save dataset
# dump(data, os.path.join(clf_dir, 'data'))

# labels = data['labels_er']
# ids = labels.index
# labels = labels['er_label'].to_numpy()

labels = pd.read_csv(os.path.join(clf_dir, 'core_labels_er.csv'), index_col=0)
ids = labels.index
avg_its = pd.read_csv(os.path.join(clf_dir, 'avg_intensities.csv'), index_col=0)

labels = labels['er_label'].to_numpy()
scores = avg_its['brown'].to_numpy()

get_roc(labels, scores, 'Average Brown Intensity', clf_dir)
