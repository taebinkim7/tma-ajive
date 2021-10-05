import pandas as pd
import os
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from patch_classifier.patches.PatchGrid import PatchGrid
from tma_ajive.Paths import Paths
from tma_ajive.utils import retain_pandas


def load_analysis_data(paths=Paths(), level='subj', load_patch_data=False):

    ##############
    # image data #
    ##############

    feats_dir = paths.features_dir
    labels_dir = paths.classification_dir

    # image features
    feats_he = pd.read_csv(os.path.join(feats_dir,
        level + '_features_he.csv'), index_col=0)
    feats_er = pd.read_csv(os.path.join(feats_dir,
        level + '_features_er.csv'), index_col=0)
    labels_er = pd.read_csv(os.path.join(labels_dir,
        level + '_labels_er.csv'), index_col=0)

    # patch data
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
    feats_he = retain_pandas(feats_he, image_feats_processor.fit_transform)
    feats_er = retain_pandas(feats_er, image_feats_processor.fit_transform)

    #############
    # alignment #
    #############
    intersection = list(set(feats_he.index)\
        .intersection(set(feats_er.index))\
        .intersection(set(labels_er.index)))
    intersection.sort()

    print('No. intersection: {}'.format(len(intersection)))

    feats_he = feats_he.loc[intersection]
    feats_er = feats_er.loc[intersection]
    labels_er = labels_er.loc[intersection]

    # sort by index
    # idx = feats_he.index.sort_values()
    # feats_he = feats_he.loc[idx]
    # feats_er = feats_er.loc[idx]
    # labels_er = labels_er.loc[idx]

    return {'feats_he': feats_he,
            'feats_er': feats_er,
            'labels_er' : labels_er,
            'image_feats_processor': image_feats_processor,
            'patch_dataset_he': patch_dataset_he,
            'patch_dataset_er': patch_dataset_er,
            'patch_feats_he': patch_feats_he,
            'patch_feats_er': patch_feats_er}
