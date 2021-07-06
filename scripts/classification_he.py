import os
import numpy as np

from joblib import dump
from tqdm import tqdm
from tma_ajive.Paths import Paths
from tma_ajive.load_image_feats import load_image_feats
from tma_ajive.classification import base_classification


data = load_image_feats(load_patch_data=False)
feats_he = data['subj_feats_he']

labels_file = os.path.join(Paths().classification_dir, 'subj_labels_er.csv')
labels = pd.read_csv(labels_file, index_col=0)

intersection = list(set(feats_he.index).intersection(set(labels.index)))
feats_he = feats_he.loc[intersection]
labels = labels.loc[intersection]

test_acc_list = []
for i in tqdm(range(100)):
    n = len(labels)
    perm_idx = np.random.RandomState(seed=None).permutation(np.arange(n))
    train_idx, test_idx = perm_idx[:int(.8 * n)], perm_idx[int(.8 * n):]

    train_feats, train_labels = feats_he.iloc[train_idx], labels.iloc[train_idx]
    test_feats, test_labels = feats_he.iloc[test_idx], labels.iloc[test_idx]

    train_dataset = [train_feats, train_labels]
    test_dataset = [test_feats, test_labels]

    _, test_acc = base_classification(train_dataset, test_dataset, 'dwd')
    test_acc_list.append(test_acc)

# dump(test_acc_list, os.path.join(Paths().classification_dir, 'test_acc_list'))
mean_test_acc = np.mean(test_acc_list)
print(mean_test_acc)
