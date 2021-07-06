import os
import pandas as pd
from joblib import dump
from jive.AJIVE import AJIVE


# initial signal ranks determined from PCA scree plots
init_signal_ranks = {'he': 50, 'er': 50, 'labels': 1}

def fit_ajive(feats_he, feats_er, labels):
    dummy = pd.concat([labels, 1 - labels], axis=1)
    ajive = AJIVE(init_signal_ranks=init_signal_ranks,
                  n_wedin_samples=1000, n_randdir_samples=1000,
                  #zero_index_names=False,
                  n_jobs=-1, store_full=False)
    ajive = ajive.fit({'he': feats_he, 'er': feats_er, 'labels': dummy})

    return ajive
