import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
from patch_classifier import WDWDClassifier
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.Paths import Paths
from tma_ajive.viz_utils import get_extreme_images

data = load_analysis_data(load_patch_data=False)

# save dataset
dump(data, os.path.join(Paths().classification_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']
ids = labels.index

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

tp_ids = ids[tp_idx][scores[tp_idx].argsort()]
fn_ids = ids[fn_idx][scores[fn_idx].argsort()]
fp_ids = ids[fp_idx][scores[fp_idx].argsort()]
tn_ids = ids[tn_idx][scores[tn_idx].argsort()]

df_tp = pd.DataFrame({'TP': tp_ids})
df_fn = pd.DataFrame({'FN': fn_ids})
df_fp = pd.DataFrame({'FP': fp_ids})
df_tn = pd.DataFrame({'TN': tn_ids})

df_all = pd.concat([df_tp, df_fn, df_fp, df_tn], axis=1)
df_all.to_csv(os.path.join(Paths().classification_dir,
              'classification_ids.csv'), index=False)
