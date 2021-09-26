import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from tma_ajive.Paths import Paths
from tma_ajive.classification import get_roc
from tma_ajive.load_analysis_data import load_analysis_data


data = load_analysis_data(load_patch_data=False)
clf_dir = Paths().classification_dir

# save dataset
# dump(data, os.path.join(clf_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']
ids = labels.index

feats = feats.to_numpy()
labels = labels['er_label'].to_numpy()

wdwd_file = os.path.join(clf_dir, 'wdwd_all')
if os.path.isfile(wdwd_file):
    # load wDWD if it exists
    classifier = WDWDClassifier.load(wdwd_file)
else:
    # train wDWD and save it
    classifier = WDWDClassifier().fit(feats, labels)
    dump(classifier, os.path.join(clf_dir, 'wdwd_all'))

scores = feats @ classifier.coef_.T + classifier.intercept_
scores = scores.reshape(-1)

get_roc(labels, scores, 'wdwd', clf_dir)
