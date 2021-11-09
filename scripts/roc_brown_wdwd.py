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


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
parser.add_argument('--iter', type=int, default=10)
args = parser.parse_args()

data_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

data = load_analysis_data(paths=paths, level=args.level)
clf_dir = paths.classification_dir

# save dataset
# dump(data, os.path.join(clf_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']
avg_its = pd.read_csv(os.path.join(clf_dir, args.level + '_avg_intensities.csv'),
                      index_col=0)
intersection = list(set(labels.index).intersection(set(avg_its.index)))
intersection.sort()

feats = feats.loc[intersection]
labels = labels.loc[intersection]
avg_its = avg_its.loc[intersection]

feats = feats.to_numpy()
labels = labels.to_numpy().reshape(-1)

# ROC using average brown intensity scores
brown_scores = avg_its['brown'].to_numpy()
brown_fpr, brown_tpr, _ = roc_curve(labels, brown_scores)
brown_auc = roc_auc_score(labels, brown_scores)

# ROC using WDWD scores
wdwd_file = os.path.join(clf_dir, args.level + '_wdwd_all')
if os.path.isfile(wdwd_file):
    # load WDWD if it exists
    classifier = WDWDClassifier.load(wdwd_file)
else:
    # train WDWD and save it
    classifier = WDWDClassifier().fit(feats, labels)
    dump(classifier, os.path.join(clf_dir, args.level + '_wdwd_all'))

wdwd_scores = feats @ classifier.coef_.T + classifier.intercept_
wdwd_scores = wdwd_scores.reshape(-1)
wdwd_fpr, wdwd_tpr, _ = roc_curve(labels, wdwd_scores)
wdwd_auc = roc_auc_score(labels, wdwd_scores)

# ROC curves
plt.plot(brown_fpr, brown_tpr, label='Avg. brown')
plt.plot(wdwd_fpr, wdwd_tpr, label='WDWD')
plt.title('ROC of Avg. brown (AUC: {}) & WDWD (AUC: {})'\
    .format(round(brown_auc, 3), round(wdwd_auc, 3)))
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.legend(loc='lower right')

# save plot
plt.savefig(os.path.join(clf_dir, 'roc_brown_wdwd.png'))
