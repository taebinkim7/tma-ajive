import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from argparse import ArgumentParser
from joblib import dump
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
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
data = load_analysis_data(paths=paths, level='core')
clf_dir = paths.classification_dir

# save dataset
# dump(data, os.path.join(clf_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']
avg_its = pd.read_csv(os.path.join(clf_dir, 'core_avg_intensities.csv'),
                      index_col=0)
intersection = list(set(labels.index).intersection(set(avg_its.index)))
intersection.sort()

feats = feats.loc[intersection]
labels = labels.loc[intersection]
avg_its = avg_its.loc[intersection]

feats = feats.to_numpy()
labels = labels['er_label'].to_numpy()

# get brown intensity scores
brown_scores = avg_its['brown'].to_numpy()

# get WDWD scores
wdwd_file = os.path.join(clf_dir, 'core_wdwd_all')
if os.path.isfile(wdwd_file):
    # load WDWD if it exists
    classifier = WDWDClassifier.load(wdwd_file)
else:
    # train WDWD and save it
    classifier = WDWDClassifier().fit(feats, labels)
    dump(classifier, os.path.join(clf_dir, 'core_wdwd_all'))

wdwd_scores = feats @ classifier.coef_.T + classifier.intercept_
wdwd_scores = wdwd_scores.reshape(-1)

# brown score vs WDWD score
neg_idx = (labels == 0)
pos_idx = (labels == 1)
plt.scatter(wdwd_scores[neg_idx], brown_scores[neg_idx], c='blue', s=3,
            alpha=.3, label='neg')
plt.scatter(wdwd_scores[pos_idx], brown_scores[pos_idx], c='red', s=3,
            alpha=.3, label='pos')
plt.title('Avg. brown vs. WDWD score')
plt.xlabel('WDWD score')
plt.ylabel('Avg. brown')
plt.legend(loc='upper left')

# save plot
plt.savefig(os.path.join(clf_dir, 'brown_wdwd.png'))
