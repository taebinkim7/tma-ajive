# for 9741 only
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from tma_ajive.Paths import Paths
from tma_ajive.classification import get_roc


clf_dir = Paths().classification_dir
df = pd.read_csv(os.path.join(clf_dir, 'subj_er_esr1.csv'),
                 index_col=0)
df = df.to_numpy()

labels, scores = df[:, 0], df[:, 1]

get_roc(labels, scores, 'esr1', clf_dir)
