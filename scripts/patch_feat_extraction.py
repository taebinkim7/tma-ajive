import os

from argparse import ArgumentParser
from patch_classifier.patches.patch_feat_extraction import patch_feat_extraction
from tma_ajive.Paths import Paths


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--image_types', type=str, nargs='+', default=['he', 'er'])
args = parser.parse_args()

data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

for type in args.image_types:
    patch_feat_extraction(paths, type)

# patch_feat_extraction(paths, 'he')
# patch_feat_extraction(paths, 'er')
