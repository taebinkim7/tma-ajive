import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from glob import glob
from math import ceil
from skimage.io import imread
from tma_ajive.Paths import Paths


def savefig(fpath, dpi=100):
    """
    Save and close a figure.
    """
    plt.savefig(fpath, bbox_inches='tight', frameon=False, dpi=dpi)
    plt.close()


def mpl_noaxis(labels=False):
    """
    Do not display any axes for any figure.
    """

    mpl.rcParams['axes.linewidth'] = 0

    if not labels:
        mpl.rcParams['xtick.bottom'] = False
        mpl.rcParams['xtick.labelbottom'] = 0

        mpl.rcParams['ytick.left'] = False
        mpl.rcParams['ytick.labelleft'] = 0


def get_extreme_images(ids, image_type, save_dir, n_subjects=9, plot_all=True):
    os.makedirs(save_dir, exist_ok=True)
    n_subjects = min(n_subjects, len(ids) // 2)
    left_ext_ids, right_ext_ids = ids[:n_subjects], ids[-n_subjects:]

    left_file = os.path.join(save_dir, 'left')
    right_file = os.path.join(save_dir, 'right')
    left_all_file = os.path.join(save_dir, 'left_all')
    right_all_file = os.path.join(save_dir, 'right_all')

    plot_images(left_ext_ids, image_type, left_file)
    plot_images(right_ext_ids, image_type, right_file)
    plot_all_images(left_ext_ids, image_type, left_file)
    plot_all_images(right_ext_ids, image_type, right_file)


def plot_all_images(ids, image_type, save_file):
    n = len(ids)
    # generate h x 3 subplot grid
    fig, axs = plt.subplots(nrows=3, ncols=n, figsize=(5 * n, 5 * 3))
    for i in range(n):
        files = glob(os.path.join(Paths().images_dir,
                                  image_type.lower(),
                                  ids[i] + '_core*'))
        for j, file in enumerate(files):
            ax = axs[j, i]
            img = imread(file)
            ax.imshow(img)
            ax.set_xlabel('{}'.format(ids[i]), fontsize=20)
            ax.tick_params(top=False, bottom=False, left=False, right=False,
                           labelleft=False, labelbottom=False)
        fig.savefig(save_file)

def plot_images(ids, image_type, save_file):
    n = len(ids)
    # generate h x 3 subplot grid
    h = ceil(n / 3)
    fig, axs = plt.subplots(nrows=3, ncols=h, figsize=(10 * h, 10 * 3))
    for i, ax in enumerate(axs.flat):
        if i >= n:
            continue
        file = glob(os.path.join(Paths().images_dir,
                                 image_type.lower(),
                                 ids[i] + '_core*'))[0]
        img = imread(file)
        ax.imshow(img)
        ax.set_xlabel('{}'.format(ids[i]), fontsize=40)
        ax.tick_params(top=False, bottom=False, left=False, right=False,
                       labelleft=False, labelbottom=False)
    fig.savefig(save_file)
