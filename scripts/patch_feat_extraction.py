import os

from argparse import ArgumentParser
from patch_classifier.patches.patch_feat_extraction import patch_feat_extraction
from tma_ajive.Paths import Paths


parser = ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--patch_size', type=int, default=200)
parser.add_argument('--model', type=str, default='vgg16')
parser.add_argument('--image_types', type=str, nargs='+', default=['he', 'er'])
args = parser.parse_args()

data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
paths = Paths(data_dir)

for type in args.image_types:
    patch_feat_extraction(paths=paths,
                          image_type=type,
                          patch_size=args.patch_size,
                          pretrained_model=args.model)
