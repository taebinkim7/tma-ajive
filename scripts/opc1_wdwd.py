import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from joblib import dump
from sklearn.decomposition import PCA
from patch_classifier import WDWDClassifier
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.Paths import Paths
from tma_ajive.viz_utils import get_extreme_images


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
args = parser.parse_args()

data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

data = load_analysis_data(paths=paths, level=args.level)

# save dataset
dump(data, os.path.join(Paths().classification_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']
ids = labels.index

feats = feats.to_numpy()
labels = labels['er_label'].to_numpy()

wdwd_file = os.path.join(Paths().classification_dir, args.level + '_wdwd_all')
if os.path.isfile(wdwd_file):
    # load WDWD if it exists
    wdwd = WDWDClassifier.load(wdwd_file)
else:
    # train WDWD and save it
    wdwd = WDWDClassifier().fit(feats, labels)
    wdwd.save(wdwd_file)

# define variables for visualization
wdwd_preds = wdwd.predict(feats)
wdwd_scores = feats @ wdwd.coef_.T + wdwd.intercept_
wdwd_scores = scores.reshape(-1)

# calculate orthogonal PC scores
feats1 = feats - feats @ wdwd.coef_.T @ wdwd.coef_
pca1 = PCA().fit(feats1)
opc1 = pca1.components_[0]
opc1_scores = feats @ opc1

# plot scores
plt.scatter(wdwd_scores, opc1_scores, s=3, alpha=.3)
plt.title('OPC1 score vs. WDWD score')
plt.xlabel('WDWD score')
plt.ylabel('OPC1 score')

# save plot
plt.savefig(os.path.join(paths.classification_dir, 'opc1_wdwd.png'))
