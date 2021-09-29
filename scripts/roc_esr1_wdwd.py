# for 9741 only
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
data = load_analysis_data(paths=paths, level='subj')
clf_dir = paths.classification_dir

# save dataset
# dump(data, os.path.join(clf_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']
esr1 = pd.read_csv(os.path.join(clf_dir, 'subj_er_esr1.csv'), index_col=0)

intersection = list(set(labels.index).intersection(set(esr1.index)))
intersection.sort()

feats = feats.loc[intersection]
labels = labels.loc[intersection]
esr1 = esr1.loc[intersection]

feats = feats.to_numpy()
labels = labels['er_label'].to_numpy()

# ROC using average ESR1
esr1_scores = esr1['esr1'].to_numpy()
esr1_fpr, esr1_tpr, _ = roc_curve(labels, esr1_scores)
esr1_auc = roc_auc_score(labels, esr1_scores)

df = pd.read_csv(os.path.join(clf_dir, 'subj_er_esr1.csv'),
                 index_col=0)
df = df.to_numpy()

labels, scores = df[:, 0], df[:, 1]


# ROC using WDWD scores
wdwd_file = os.path.join(clf_dir, 'subj_wdwd_all')
if os.path.isfile(wdwd_file):
    # load WDWD if it exists
    classifier = WDWDClassifier.load(wdwd_file)
else:
    # train WDWD and save it
    classifier = WDWDClassifier().fit(feats, labels)
    dump(classifier, os.path.join(clf_dir, 'subj_wdwd_all'))

wdwd_scores = feats @ classifier.coef_.T + classifier.intercept_
wdwd_scores = wdwd_scores.reshape(-1)
wdwd_fpr, wdwd_tpr, _ = roc_curve(labels, wdwd_scores)
wdwd_auc = roc_auc_score(labels, wdwd_scores)

# ROC curves
plt.plot(esr1_fpr, esr1_tpr, label='ESR1')
plt.plot(wdwd_fpr, wdwd_tpr, label='WDWD')
plt.title('ROC of ESR1 (AUC: {}) & WDWD (AUC: {})'\
    .format(round(esr1_auc, 3), round(wdwd_auc, 3)))
plt.xlabel('1 - specificity')
plt.ylabel('sensitivity')
plt.legend(loc='lower right')

# save plot
plt.savefig(os.path.join(clf_dir, 'roc_esr1_wdwd.png'))
