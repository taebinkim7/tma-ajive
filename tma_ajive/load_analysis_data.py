import pandas as pd
import os
import json
import numpy as np

from sklearn.preprocessing import StandardScaler
from patch_classifier.patches.PatchGrid import PatchGrid
from tma_ajive.Paths import Paths
from tma_ajive.utils import retain_pandas


def load_analysis_data(paths=Paths(), level='subj',
                       types=['feats_he', 'feats_er', 'labels_er'],
                       load_patch_data=False):

    # define dictionary to return
    data = {}

    ##############
    # image data #
    ##############

    feats_dir = paths.features_dir
    clf_dir = paths.classification_dir

    # image features
    if 'feats_he' in types:
        feats_he = pd.read_csv(os.path.join(feats_dir,
            level + '_features_he.csv'), index_col=0)
        data['feats_he'] = feats_he
    if 'feats_er' in types:
        feats_er = pd.read_csv(os.path.join(feats_dir,
            level + '_features_er.csv'), index_col=0)
        data['feats_er'] = feats_er

    # clinical data
    if 'labels_er' in types:
        labels_er = pd.read_csv(os.path.join(clf_dir,
            level + '_labels_er.csv'), index_col=0)
        data['labels_er'] = labels_er
    if 'surv_mos' in types:
        surv_mos = pd.read_csv(os.path.join(clf_dir,
            level + '_surv_mos.csv'), index_col=0)
        data['surv_mos'] = surv_mos

    #############
    # alignment #
    #############
    index_sets = list(map(lambda x: set(x.index), list(data.values())))
    intersection = set.intersection(*index_sets)
    intersection.sort()

    print('No. intersection: {}'.format(len(intersection)))

    for type in data:
        data[type] = data[type].loc[intersection]

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
        data['patch_dataset_he'] = patch_dataset_he
        data['patch_dataset_er'] = patch_dataset_er
        data['patch_feats_he'] = patch_feats_he
        data['patch_feats_er'] = patch_feats_er

    # process data
    image_feats_processor = StandardScaler()
    data['image_feats_processor'] = image_feats_processor
    feats_he = retain_pandas(feats_he, image_feats_processor.fit_transform)
    feats_er = retain_pandas(feats_er, image_feats_processor.fit_transform)


    return data
