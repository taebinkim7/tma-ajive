# for 9741 only
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')

from argparse import ArgumentParser
from tma_ajive.Paths import Paths
from tma_ajive.classification import get_roc


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
args = parser.parse_args()

data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)
clf_dir = paths.classification_dir
df = pd.read_csv(os.path.join(clf_dir, 'subj_er_esr1.csv'),
                 index_col=0)
df = df.to_numpy()

labels, scores = df[:, 0], df[:, 1]

get_roc(labels, scores, 'ESR1', clf_dir)
