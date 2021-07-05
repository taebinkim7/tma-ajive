import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
from patch_classifier.patches.PatchGrid import PatchGrid
from tma_ajive.Paths import Paths
from tma_ajive.utils import retain_pandas, get_mismatches


def load_image_feats(load_patch_feats=False):

    ##############
    # image data #
    ##############

    feats_dir = os.path.join(Paths().features_dir)

    # image patch features
    subj_feats_he = pd.read_csv(os.path.join(feats_dir, 'subj_features_he.csv'),
                                index_col=0)
    subj_feats_er = pd.read_csv(os.path.join(feats_dir, 'subj_features_er.csv'),
                                index_col=0)
    subj_feats_he.index = subj_feats_he.index.astype(str)
    subj_feats_er.index = subj_feats_er.index.astype(str)

    if load_patch_data:
        # patches dataset
        patch_dataset_he = \
            PatchGrid.load(os.path.join(feats_dir, 'patch_dataset_he'))
        patch_dataset_er = \
            PatchGrid.load(os.path.join(feats_dir, 'patch_dataset_er'))
        patch_feats_he = \
            pd.read_csv(os.path.join(feats_dir, 'patch_features_he.csv'),
                        index_col=['image', 'patch_idx'])
        patch_feats_er = \
            pd.read_csv(os.path.join(feats_dir, 'patch_features_er.csv'),
                        index_col=['image', 'patch_idx'])
    else:
        patch_dataset_he, patch_dataset_er, patch_feats_he, patch_feats_er = \
            None, None, None, None

    # process data
    image_feats_processor = StandardScaler()
    subj_feats_he = retain_pandas(subj_feats_he,
                                  image_feats_processor.fit_transform)
    subj_feats_er = retain_pandas(subj_feats_er,
                                  image_feats_processor.fit_transform)

    #############
    # alignment #
    #############
    intersection = list(set(subj_feats_he.index)\
        .intersection(set(subj_feats_er.index)))
    he_only, er_only = get_mismatches(subj_feats_he.index, subj_feats_er.index)

    print('No. intersection: {}'.format(len(intersection)))
    print('No. only in HE, not in ER: {}'.format(len(he_only)))
    print('No. only in ER, not in HE: {}'.format(len(er_only)))

    subj_feats_he = subj_feats_he.loc[intersection]
    subj_feats_er = subj_feats_er.loc[intersection]
    subj_feats_he_only = subj_feats_he.loc[he_only]
    subj_feats_er_only = subj_feats_er.loc[er_only]

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
            'subj_feats_he_only': subj_feats_he_only,
            'subj_feats_er_only': subj_feats_er_only,
            'image_feats_processor': image_feats_processor}
