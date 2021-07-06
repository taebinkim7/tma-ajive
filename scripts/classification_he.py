import os
import numpy as np

from joblib import dump
from tqdm import tqdm
from tma_ajive.load_image_feats import load_image_feats
from tma_ajive.classification import base_classification


data = load_image_feats(load_patch_data=False)
subj_feats_he = data['subj_feats_he']
feats = subj_feats_he
# subj_feats_he_only = data['subj_feats_he_only']
# feats = subj_feats_he.append(subj_feats_he_only)

labels_file = os.path.join(Paths().classification_dir, 'subj_labels_er.csv')
labels = pd.read_csv(labels_file, index_col=0)

intersection = list(set(feats.index).intersection(set(labels.index)))
feats = feats.loc[intersection]
labels = labels.loc[intersection]

test_acc_list = []
for i in tqdm(range(100)):
    n = len(labels)
    perm_idx = np.random.RandomState(seed=seed).permutation(np.arange(n))
    train_idx, test_idx = perm_idx[:int(.8 * n)], perm_idx[int(.8 * n):]

    train_dataset = [feats.iloc[train_idx], labels.iloc[train_idx]]
    test_dataset = [feats.iloc[test_idx], labels.iloc[test_idx]]

    _, test_acc = base_classification(train_dataset, test_dataset, 'dwd')
    test_acc_list.append(test_acc)

# dump(test_acc_list, os.path.join(Paths().classification_dir, 'test_acc_list'))
mean_test_acc = np.mean(test_acc_list)
print(mean_test_acc)
