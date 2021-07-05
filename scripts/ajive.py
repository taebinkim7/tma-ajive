import os
from joblib import dump
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from jive.AJIVE import AJIVE

from tma_ajive.load_image_feats import load_image_feats
from tma_ajive.viz_utils import savefig, mpl_noaxis
from tma_ajive.Paths import Paths


# make directories for saved results
os.makedirs(os.path.join(Paths().results_dir, 'data'), exist_ok=True)
os.makedirs(os.path.join(Paths().results_dir, 'common', 'loadings'), exist_ok=True)
os.makedirs(os.path.join(Paths().results_dir, 'he_indiv', 'loadings'), exist_ok=True)
os.makedirs(os.path.join(Paths().results_dir, 'er_indiv', 'loadings'), exist_ok=True)


# load pre-computed data e.g. patch features
data = load_image_feats(load_patch_feats=False)
subj_feats_he = data['subj_feats_he']
subj_feats_er = data['subj_feats_er']

# initial signal ranks determined from PCA scree plots
init_signal_ranks = {'he': 40, 'er': 40}

# run AJIVE
ajive = AJIVE(init_signal_ranks=init_signal_ranks,
              n_wedin_samples=1000, n_randdir_samples=1000,
              #zero_index_names=False,
              n_jobs=-1, store_full=False)
ajive = ajive.fit({'he': subj_feats_he, 'er': subj_feats_er})

dump(ajive, os.path.join(Paths().results_dir, 'data', 'fit_ajive'))

# #####################
# # AJIVE diagnostics #
# #####################
#
# # diagnostic plot
# plt.figure(figsize=[10, 10])
# ajive.plot_joint_diagnostic()
# savefig(os.path.join(Paths().results_dir, 'ajive_diagnostic.png'))
#
# #################
# # plot loadings #
# #################
#
# # set visualization configs
# mpl_noaxis(labels=True)
#
# n_genes = 90
# inches = 5
# height_scale = n_genes // 25
# load_figsize = (inches, height_scale * inches)
#
# # common loadings
# load_dir = os.path.join(Paths().results_dir, 'common', 'loadings')
# os.makedirs(load_dir, exist_ok=True)
# for r in range(ajive.common.rank):
#     plt.figure(figsize=load_figsize)
#     plt.plot(ajive.common.loadings(r))
#     #ajive.blocks['he'].plot_common_loading(r)
#     plt.title('common component {}'.format(r + 1))
#     savefig(os.path.join(load_dir, 'loadings_comp_{}.png'.format(r + 1)))
#
#
# # he individual loadings
# load_dir = os.path.join(Paths().results_dir, 'he_indiv', 'loadings')
# os.makedirs(load_dir, exist_ok=True)
# n_indiv_comps = min(5, ajive.blocks['he'].individual.rank)
# for r in range(n_indiv_comps):
#     plt.figure(figsize=load_figsize)
#     plt.plot(ajive.blocks['he'].individual.loadings(r))
#     plt.title('HE individual component {}'.format(r + 1))
#     savefig(os.path.join(load_dir, 'loadings_comp_{}.png'.format(r + 1)))
#
# # er individual loadings
# load_dir = os.path.join(Paths().results_dir, 'er_indiv', 'loadings')
# os.makedirs(load_dir, exist_ok=True)
# n_indiv_comps = min(5, ajive.blocks['er'].individual.rank)
# for r in range(n_indiv_comps):
#     plt.figure(figsize=load_figsize)
#     plt.plot(ajive.blocks['er'].individual.loadings(r))
#     plt.title('ER individual component {}'.format(r + 1))
#     savefig(os.path.join(load_dir, 'loadings_comp_{}.png'.format(r + 1)))
