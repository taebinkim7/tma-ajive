import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
from skimage.io import imread
from patch_classifier import DWDClassifier
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.classification import get_train_test_ids
from tma_ajive.Paths import Paths

data = load_analysis_data(load_patch_data=False)

# save dataset
dump(data, os.path.join(Paths().classification_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']

# get balanced dataset to train classifier
bal_ids = get_train_test_ids(labels, balanced=True)
bal_feats = feats.loc[bal_ids]
bal_labels = labels.loc[bal_ids]

bal_feats = bal_feats.to_numpy()
bal_labels = bal_labels['er_label'].to_numpy()

dwd_file = os.path.join(Paths().classification_dir, 'dwd_all')
if os.path.isfile(dwd_file):
    # load DWD if it exists
    dwd = DWDClassifier.load(dwd_file)
else:
    # train DWD and save it
    dwd = DWDClassifier().fit(bal_feats, bal_labels)
    dump(dwd, os.path.join(Paths().classification_dir, 'dwd_all'))

# define variables for visualization
feats = feats.to_numpy()
labels = labels['er_label'].to_numpy()
preds = dwd.predict(feats)
scores = feats @ dwd.coef_.T
scores = scores.reshape(-1)

tp_idx = (labels == 1) & (preds == 1)
fn_idx = (labels == 1) & (preds == 0)
fp_idx = (labels == 0) & (preds == 1)
tn_idx = (labels == 0) & (preds == 0)

# make directory to save plots
plot_dir = os.path.join(Paths().classification_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# plot DWD scores
n = len(ids)
noise = np.random.RandomState(10).normal(scale=.1, size=n)
ax = plt.axes()
ax.set_title('Scatterplot of DWD scores (jitter = N(0, 0.1))', fontsize=13, fontweight='bold')
ax.scatter(scores[tp_idx | fn_idx], labels[tp_idx | fn_idx] + noise[tp_idx | fn_idx], c='red', s=3, label='pos')
ax.scatter(scores[fp_idx | tn_idx], labels[fp_idx | tn_idx] + noise[fp_idx | tn_idx], c='blue', s=3, label='neg')
ax.set_xlabel('Score')
ax.set_ylabel('ER Label')

ax1 = ax.twinx()
sns.kdeplot(x=scores[tp_idx | fn_idx], ax=ax1, c='red')
sns.kdeplot(x=scores[fp_idx | tn_idx], ax=ax1, c='blue')

ax.legend(loc='lower right')

ax.savefig(os.path.join(plot_dir, 'dwd_scores.png'))

# mark misclassified objects
n = len(ids)
noise = np.random.RandomState(10).normal(scale=.1, size=n)
ax = plt.axes()
ax.set_title('Scatterplot of DWD scores (jitter = N(0, 0.1))', fontsize=13, fontweight='bold')
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

ax.savefig(os.path.join(plot_dir, 'dwd_scores_misclf.png'))

# extreme images
tp_ids = ids[tp_idx][scores[tp_idx].argsort()]
fn_ids = ids[fn_idx][scores[fn_idx].argsort()]
fp_ids = ids[fp_idx][scores[fp_idx].argsort()]
tn_ids = ids[tn_idx][scores[tn_idx].argsort()]

get_extreme_images(tp_ids)
get_extreme_images(fn_ids)
get_extreme_images(fp_ids)
get_extreme_images(tn_ids)


def get_extreme_images(ids, save_dir, n_subjects=9):
    left_ext_ids, right_ext_ids = ids[:n_subjects], ids[-n_subjects:]
    plot_images(left_ext_ids)


def plot_images(ids, save_name):
    n = len(ids)
    # generate h x 3 subplot grid
    h = n // 3
    fig, axs = plt.subplots(3, h, figsize=(10 * 3, 10 * h))
    for i, ax in enumerate(axs.flat):
        if i >= n:
            continue
        file = glob(os.path.join(Paths().images_dir,
                                 image_type.lower(),
                                 ids[i] + '_core*'))[0]
        img = imread(file)
        ax.imshow(img)
        ax.set_xlabel('{}'.format(ids[i]))
    fig.savefig(save_name)
