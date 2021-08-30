import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
from skimage.io import imread
from patch_classifier import WDWDClassifier
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.Paths import Paths
from tma_ajive.viz_utils import get_extreme_images

data = load_analysis_data(load_patch_data=False)

# save dataset
dump(data, os.path.join(Paths().classification_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']

feats = feats.to_numpy()
labels = labels['er_label'].to_numpy()

wdwd_file = os.path.join(Paths().classification_dir, 'wdwd_all')
if os.path.isfile(wdwd_file):
    # load wDWD if it exists
    classifier = WDWDClassifier.load(wdwd_file)
else:
    # train wDWD and save it
    classifier = WDWDClassifier().fit(feats, labels)
    dump(classifier, os.path.join(Paths().classification_dir, 'wdwd_all'))

# define variables for visualization
preds = classifier.predict(feats)
scores = feats @ classifier.coef_.T + classifier.intercept_
scores = scores.reshape(-1)

tp_idx = (labels == 1) & (preds == 1)
fn_idx = (labels == 1) & (preds == 0)
fp_idx = (labels == 0) & (preds == 1)
tn_idx = (labels == 0) & (preds == 0)

# make directory to save plots
plot_dir = os.path.join(Paths().classification_dir, 'wdwd_plots')
os.makedirs(plot_dir, exist_ok=True)

# plot wDWD scores
ids = labels.index
n = len(ids)
noise = (np.arange(n) - n // 2) / 5000
ax = plt.axes()
ax.set_title('Scatterplot of wDWD scores (jitter = ID)', fontsize=13, fontweight='bold')
ax.scatter(scores[tp_idx | fn_idx], labels[tp_idx | fn_idx] + noise[tp_idx | fn_idx], c='red', s=3, label='pos')
ax.scatter(scores[fp_idx | tn_idx], labels[fp_idx | tn_idx] + noise[fp_idx | tn_idx], c='blue', s=3, label='neg')
ax.set_xlabel('Score')
ax.set_ylabel('ER Label')
ax1 = ax.twinx()
sns.kdeplot(x=scores[tp_idx | fn_idx], ax=ax1, c='red')
sns.kdeplot(x=scores[fp_idx | tn_idx], ax=ax1, c='blue')
ax.legend(loc='lower right')
ax.savefig(os.path.join(plot_dir, 'wdwd_scores.png'))

# mark misclassified objects
ax = plt.axes()
ax.set_title('Scatterplot of wDWD scores (jitter = ID)', fontsize=13, fontweight='bold')
ax.scatter(scores[tp_idx], labels[tp_idx] + noise[tp_idx], c='red', s=3, label='tp')
ax.scatter(scores[tn_idx], labels[tn_idx] + noise[tn_idx], c='blue', s=3, label='tn')
ax.scatter(scores[fn_idx], labels[fn_idx] + noise[fn_idx], c='green', s=3, label='fn')
ax.scatter(scores[fp_idx], labels[fp_idx] + noise[fp_idx], c='orange', s=3, label='fp')
ax.set_xlabel('Score')
ax.set_ylabel('ER Label')
ax1 = ax.twinx()
sns.kdeplot(x=scores[tp_idx | fn_idx], ax=ax1, c='red')
sns.kdeplot(x=scores[fp_idx | tn_idx], ax=ax1, c='blue')
ax.axvline((min(scores[tp_idx]) + max(scores[tn_idx])) / 2, c='black', ls='--')
ax.legend(loc='lower right')
ax.savefig(os.path.join(plot_dir, 'wdwd_scores_misclf.png'))

# extreme images
tp_ids = ids[tp_idx][scores[tp_idx].argsort()]
fn_ids = ids[fn_idx][scores[fn_idx].argsort()]
fp_ids = ids[fp_idx][scores[fp_idx].argsort()]
tn_ids = ids[tn_idx][scores[tn_idx].argsort()]

tp_dir = os.path.join(plot_dir, 'tp')
fn_dir = os.path.join(plot_dir, 'fn')
fp_dir = os.path.join(plot_dir, 'fp')
tn_dir = os.path.join(plot_dir, 'tn')

get_extreme_images(tp_ids, tp_dir)
get_extreme_images(fn_ids, fn_dir)
get_extreme_images(fp_ids, fp_dir)
get_extreme_images(tn_ids, tn_dir)
