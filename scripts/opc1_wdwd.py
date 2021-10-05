import os
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import torch

from argparse import ArgumentParser
from joblib import dump
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from patch_classifier import WDWDClassifier
from tma_ajive.load_analysis_data import load_analysis_data
from tma_ajive.Paths import Paths
from tma_ajive.viz_utils import get_extreme_images
from tma_ajive.nn_classification import nn_classification, GetDataset


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--level', type=str, default='subj')
args = parser.parse_args()

data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

data = load_analysis_data(paths=paths, level=args.level)

# save dataset
# dump(data, os.path.join(paths.classification_dir, 'data'))

feats = data['feats_er']
labels = data['labels_er']
ids = labels.index

feats = feats.to_numpy()
labels = labels['er_label'].to_numpy()

wdwd_file = os.path.join(Paths().classification_dir, args.level + '_wdwd_all')
if os.path.isfile(wdwd_file):
    # load WDWD if it exists
    wdwd = WDWDClassifier.load(wdwd_file)
else:
    # train WDWD and save it
    wdwd = WDWDClassifier().fit(feats, labels)
    wdwd.save(wdwd_file)

# define variables for visualization
wdwd_preds = wdwd.predict(feats)
wdwd_scores = feats @ wdwd.coef_.T + wdwd.intercept_
wdwd_scores = wdwd_scores.reshape(-1)

# calculate orthogonal PC scores
feats1 = feats - feats @ wdwd.coef_.T @ wdwd.coef_
pca1 = PCA().fit(feats1)
opc1 = pca1.components_[0]
opc1_scores = feats @ opc1

# plot scores colored by labels
pos_idx = (labels == 0)
neg_idx = (labels == 1)
plt.scatter(wdwd_scores[pos_idx], opc1_scores[pos_idx], c='red', s=3,
            alpha=.3, label='pos')
plt.scatter(wdwd_scores[neg_idx], opc1_scores[neg_idx], c='blue', s=3,
            alpha=.3, label='neg')
plt.axvline(x=0, alpha=.5, color='black')
plt.title('OPC1 score vs. WDWD score (color = label)')
plt.xlabel('WDWD score')
plt.ylabel('OPC1 score')
plt.legend(loc='upper left')

# save plot
plt.savefig(os.path.join(paths.classification_dir, 'opc1_wdwd.png'))
plt.close()

# plot scores with nn predictions
model = nn_classification(feats, labels, model_type='mlp', p_train=.9,
                          return_model=True)
dataset = GetDataset(feats, labels, 'mlp')
loader = DataLoader(dataset, batch_size=1)

y_pred_list = []
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    for X, _ in loader:
        X = X.to(device)
        y_pred = model(X)
        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.round(y_pred)
        y_pred_list.append(y_pred.cpu().numpy())

preds = np.array([a.squeeze().tolist() for a in y_pred_list])

# plot scores colored by nn predictions
pos_idx = (preds == 0)
neg_idx = (preds == 1)
plt.scatter(wdwd_scores[pos_idx], opc1_scores[pos_idx], c='red', s=3,
            alpha=.3, label='pos')
plt.scatter(wdwd_scores[neg_idx], opc1_scores[neg_idx], c='blue', s=3,
            alpha=.3, label='neg')
plt.axvline(x=0, alpha=.5, color='black')
plt.title('OPC1 score vs. WDWD score (color = MLP prediction)')
plt.xlabel('WDWD score')
plt.ylabel('OPC1 score')
plt.legend(loc='upper left')

# save plot
plt.savefig(os.path.join(paths.classification_dir, 'opc1_wdwd_nn.png'))
