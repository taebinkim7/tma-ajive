import pandas as pd
import os
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
from patch_classifier.patches.PatchGrid import PatchGrid
from tma_ajive.Paths import Paths
from tma_ajive.utils import retain_pandas


def load_analysis_data(load_patch_data=False):

    ##############
    # image data #
    ##############

    feats_dir = Paths().features_dir
    labels_dir = Paths().classification_dir

    # image patch features
    subj_feats_he = pd.read_csv(os.path.join(feats_dir, 'subj_features_he.csv'),
                                index_col=0)
    subj_feats_er = pd.read_csv(os.path.join(feats_dir, 'subj_features_er.csv'),
                                index_col=0)
    subj_labels_er = pd.read_csv(os.path.join(labels_dir, 'subj_labels_er.csv'),
                                 index_col=0)
    subj_feats_he.index = subj_feats_he.index.astype(str)
    subj_feats_er.index = subj_feats_er.index.astype(str)
    subj_labels_er.index = subj_labels_er.index.astype(str)

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
        .intersection(set(subj_feats_er.index))\
        .intersection(set(subj_labels_er.index)))
    intersection.sort()

    print('No. intersection: {}'.format(len(intersection)))

    subj_feats_he = subj_feats_he.loc[intersection]
    subj_feats_er = subj_feats_er.loc[intersection]
    subj_labels_er = subj_labels_er.loc[intersection]

    # make sure subjects are in same order
    # idx = subj_feats_he.index.sort_values()
    # subj_feats_he = subj_feats_he.loc[idx]
    # subj_feats_er = subj_feats_er.loc[idx]
    # subj_labels_er = subj_labels_er.loc[idx]

    return {'patch_dataset_he': patch_dataset_he,
            'patch_dataset_er': patch_dataset_er,
            'patch_feats_he': patch_feats_he,
            'patch_feats_er': patch_feats_er,
            'subj_feats_he': subj_feats_he,
            'subj_feats_er': subj_feats_er,
            'subj_labels_er' : subj_labels_er,
            'image_feats_processor': image_feats_processor}
