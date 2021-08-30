import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from glob import glob

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


def get_extreme_images(ids, save_dir, n_subjects=9):
    left_ext_ids, right_ext_ids = ids[:n_subjects], ids[-n_subjects:]
    left_file = os.path.join(save_dir, 'left')
    right_file = os.path.join(save_dir, 'right')
    plot_images(left_ext_ids, left_file)
    plot_images(right_ext_ids, right_file)


def plot_images(ids, save_file):
    n = len(ids)
    # generate h x 3 subplot grid
    h = n // 3
    fig, axs = plt.subplots(3, h, figsize=(10 * 3, 10 * h))
    for i, ax in enumerate(axs.flat):
        if i >= n:
            continue
        file = glob(os.path.join(Paths().images_dir,
                                 image_type.lower(),
                                 ids[i] + '_core*'))[0]
        img = imread(file)
        ax.imshow(img)
        ax.set_xlabel('{}'.format(ids[i]))
    fig.savefig(save_file)
