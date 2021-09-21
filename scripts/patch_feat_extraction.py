import torch
import os
import numpy as np
import pandas as pd

from torchvision.transforms import Normalize, ToTensor, Compose
from patch_classifier.Paths import Paths
from patch_classifier.patches.PatchGrid import PatchGrid
from patch_classifier.patches.patch_features import compute_patch_features
from patch_classifier.patches.cnn_models import load_cnn_model
from tma_ajive.Paths import Paths as CurrPaths


# unify Paths
paths = Paths(CurrPaths().data_dir)
os.makedirs(paths.features_dir, exist_ok=True)

# CNN feature extraction model
model = load_cnn_model()

#######################
# get patches dataset #
#######################

# compute the backgorund mask for each image, break into patches, throw out
# patches which have too much background

def patch_feat_extraction(image_type):

    patch_kws = {'patch_size': 200,
                 'pad_image': 'div_200',
                 'max_prop_background': .9, # 1. for 9344
                 'threshold_algo': 'triangle_otsu',
                 'image_type': image_type}

    patch_dataset = PatchGrid(**patch_kws)
    patch_dataset.make_patch_grid()
    patch_dataset.compute_pixel_stats(image_limit=10)
    patch_dataset.save(os.path.join(paths.features_dir,
                                    'patch_dataset_' + image_type))

    ##############################
    # Extract patch CNN features #
    ##############################

    # patch image processing
    # center and scale channels
    channel_avg = patch_dataset.pixel_stats_['avg'] / 255
    channel_std = np.sqrt(patch_dataset.pixel_stats_['var']) / 255
    patch_transformer = Compose([ToTensor(),
                                 Normalize(mean=channel_avg, std=channel_std)])

    fpath = os.path.join(paths.features_dir,
                         'patch_features_' + image_type + '.csv')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    compute_patch_features(image_type=image_type,
                           patch_dataset=patch_dataset,
                           model=model,
                           fpath=fpath,
                           patch_transformer=patch_transformer,
                           device=device)


    ######################
    # save core features #
    ######################

    patch_feats = pd.read_csv(fpath, index_col=['image', 'patch_idx'])
    patch_feats_ = patch_feats.copy()
    core_feats = patch_feats_.groupby('image').mean()
    core_ids = []
    core_ids_ = np.unique(patch_feats.index.get_level_values('image'))
    for id in core_ids_:
        core_ids.append(id.split('_')[0] + '_' + id.split('_')[1])
    core_feats.index = core_ids
    core_feats.to_csv(os.path.join(paths.features_dir,
                                   'core_features_' + image_type + '.csv'))

    #########################
    # save subject features #
    #########################

    core_feats_ = core_feats.copy()
    subj_ids = []
    for id in core_ids:
        subj_ids.append(id.split('_')[0])
    # core_feats_.loc[:, 'subject'] = subj_ids
    core_feats_['subject'] = subj_ids
    subj_ids = np.unique(subj_ids)
    subj_feats = core_feats_.groupby('subject').mean()
    subj_feats.index = subj_ids
    subj_feats.to_csv(os.path.join(paths.features_dir,
                                   'subj_features_' + image_type + '.csv'))



patch_feat_extraction('he')
patch_feat_extraction('er')
