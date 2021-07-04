import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import numpy as np

from patch_classifier.patches.PatchGrid import PatchGrid

from tma_ajive.Paths import Paths
from tma_ajive.utils import retain_pandas, get_mismatches


def load_image_feats(load_patch_feats=True):

    ##############
    # image data #
    ##############

    # patches dataset
    patch_data_dir = os.path.join(Paths().features_dir)
    patch_dataset_he = PatchGrid.load(os.path.join(patch_data_dir, 'patch_dataset_he'))
    patch_dataset_er = PatchGrid.load(os.path.join(patch_data_dir, 'patch_dataset_er'))

    # image patch features
    subj_feats_he = pd.read_csv(os.path.join(patch_data_dir, 'subj_features_he.csv'),
                                    index_col=0)
    subj_feats_er = pd.read_csv(os.path.join(patch_data_dir, 'subj_features_er.csv'),
                                    index_col=0)
    subj_feats_he.index = subj_feats_he.index.astype(str)
    subj_feats_er.index = subj_feats_er.index.astype(str)
    subj_feats_he.index = [idx.split('_he')[0] for idx in subj_feats_he.index]
    subj_feats_er.index = [idx.split('_er')[0] for idx in subj_feats_er.index]

    if load_patch_feats:
        patch_feats_he = \
            pd.read_csv(os.path.join(patch_data_dir, 'patch_features_he.csv'),
                        index_col=['image', 'patch_idx'])
        patch_feats_er = \
            pd.read_csv(os.path.join(patch_data_dir, 'patch_features_er.csv'),
                        index_col=['image', 'patch_idx'])
    else:
        patch_feats_he, patch_feats_er = None, None

    #############
    # alignment #
    #############
    intersection = list(set(subj_feats_he.index).intersection(set(subj_feats_er.index)))
    in_he, in_er = get_mismatches(subj_feats_he.index, subj_feats_er.index)

    print('intersection: {}'.format(len(intersection)))
    print('in HE, not in ER: {}'.format(len(in_he)))
    print('in ER, not in HE: {}'.format(len(in_er)))

    subj_feats_he = subj_feats_he.loc[intersection]
    subj_feats_er = subj_feats_er.loc[intersection]

    print(subj_feats_he.shape)
    print(subj_feats_er.shape)

    # process data
    image_feats_processor = StandardScaler()
    subj_feats_he = retain_pandas(subj_feats_he, image_feats_processor.fit_transform)
    subj_feats_er = retain_pandas(subj_feats_er, image_feats_processor.fit_transform)

    # make sure subjects are in same order
    idx = subj_feats_he.index.sort_values()
    subj_feats_he = subj_feats_he.loc[idx]
    subj_feats_er = subj_feats_er.loc[idx]

    return {'patch_dataset_he': patch_dataset_he,
            'patch_dataset_er': patch_dataset_er,
            'patch_feats_he': patch_feats_he,
            'patch_feats_er': patch_feats_er,
            'subj_feats_he': subj_feats_he,
            'subj_feats_er': subj_feats_er,
            'image_feats_processor': image_feats_processor}


def sphere(X):
    s = 1.0 / np.array(X).std(axis=1)
    return np.array(X) * s[:, None]
