import os
import numpy as np

from joblib import dump
from tqdm import tqdm
from tma_ajive.load_image_feats import load_image_feats
from tma_ajive.clf_utils import base_classification


data = load_image_feats(load_patch_data=False)
subj_feats_he = data['subj_feats_he']
subj_feats_he_only = data['subj_feats_he_only']
feats = subj_feats_he.append(subj_feats_he_only)

test_acc_list = []
for i in tqdm(range(100)):
    _, test_acc = base_classification(feats, 'dwd')
    test_acc_list.append(test_acc)

# dump(test_acc_list, os.path.join(Paths().classification_dir, 'test_acc_list'))
mean_test_acc = np.mean(test_acc_list)
print(mean_test_acc)
