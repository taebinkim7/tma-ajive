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


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
parser.add_argument('--iter', type=int, default=10)
args = parser.parse_args()

data_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

# data = load_analysis_data(paths=paths, level=args.level)
clf_dir = paths.classification_dir

# save dataset
# dump(data, os.path.join(clf_dir, 'data'))

# labels = data['labels_er']
# ids = labels.index
# labels = labels.to_numpy().reshape(-1)

labels = pd.read_csv(os.path.join(clf_dir, args.level + '_labels_er.csv'), index_col=0)
ids = labels.index
avg_its = pd.read_csv(os.path.join(clf_dir, args.level + '_avg_intensities.csv'),
                      index_col=0)

intersection = list(set(labels.index).intersection(set(avg_its.index)))
intersection.sort()

print('No. intersection: {}'.format(len(intersection)))

labels = labels.loc[intersection]
avg_its = avg_its.loc[intersection]

labels = labels.to_numpy().reshape(-1)
scores = avg_its['brown'].to_numpy()

get_roc(labels, scores, 'Average Brown Intensity', clf_dir)
