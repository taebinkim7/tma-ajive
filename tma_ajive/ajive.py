import os
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from joblib import dump
from jive.AJIVE import AJIVE
from tma_ajive.Paths import Paths
from tma_ajive.viz_utils import savefig, mpl_noaxis

# initial signal ranks determined from PCA scree plots
init_signal_ranks = {'he': 50, 'er': 50}

def fit_ajive(feats_he, feats_er, labels=None, save=False):
    # dummy = pd.concat([labels, 1 - labels], axis=1)
    ajive = AJIVE(init_signal_ranks=init_signal_ranks,
                  n_wedin_samples=1000, n_randdir_samples=1000,
                  #zero_index_names=False,
                  n_jobs=-1, store_full=False)
    ajive = ajive.fit({'he': feats_he, 'er': feats_er})

    if save:
        dump(ajive, os.path.join(Paths().results_dir, 'data', 'fit_ajive'))

        #####################
        # AJIVE diagnostics #
        #####################

        # diagnostic plot
        plt.figure(figsize=[10, 10])
        ajive.plot_joint_diagnostic()
        savefig(os.path.join(Paths().results_dir, 'ajive_diagnostic.png'))

        #################
        # plot loadings #
        #################

        # set visualization configs
        mpl_noaxis(labels=True)

        n_genes = 90
        inches = 5
        height_scale = n_genes // 25
        load_figsize = (inches, height_scale * inches)

        # common loadings
        load_dir = os.path.join(Paths().results_dir, 'common', 'loadings')
        os.makedirs(load_dir, exist_ok=True)
        for r in range(ajive.common.rank):
            plt.figure(figsize=load_figsize)
            plt.plot(ajive.common.loadings(r))
            #ajive.blocks['he'].plot_common_loading(r)
            plt.title('common component {}'.format(r + 1))
            savefig(os.path.join(load_dir,
                                 'loadings_comp_{}.png'.format(r + 1)))


        # he individual loadings
        load_dir = os.path.join(Paths().results_dir, 'he_indiv', 'loadings')
        os.makedirs(load_dir, exist_ok=True)
        n_indiv_comps = min(5, ajive.blocks['he'].individual.rank)
        for r in range(n_indiv_comps):
            plt.figure(figsize=load_figsize)
            plt.plot(ajive.blocks['he'].individual.loadings(r))
            plt.title('HE individual component {}'.format(r + 1))
            savefig(os.path.join(load_dir,
                                 'loadings_comp_{}.png'.format(r + 1)))

        # er individual loadings
        load_dir = os.path.join(Paths().results_dir, 'er_indiv', 'loadings')
        os.makedirs(load_dir, exist_ok=True)
        n_indiv_comps = min(5, ajive.blocks['er'].individual.rank)
        for r in range(n_indiv_comps):
            plt.figure(figsize=load_figsize)
            plt.plot(ajive.blocks['er'].individual.loadings(r))
            plt.title('ER individual component {}'.format(r + 1))
            savefig(os.path.join(load_dir,
                                 'loadings_comp_{}.png'.format(r + 1)))

    return ajive
