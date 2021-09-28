##### activate conda env for stain normalization first
import os
import numpy as np
import cupy as cp
import pandas as pd

from argparse import ArgumentParser
from glob import glob
from PIL import Image
from tqdm import tqdm
from stain_norm import Normalizer
from stain_norm.utils import rgb2od, get_intensity


parser = ArgumentParser(description='Data directory')
parser.add_argument('--data_dir', type=str, action='store')
args = parser.parse_args()
data_dir = os.path.join('/datastore/nextgenout5/share/labs/smarronlab/tkim/data', args.data_dir)
er_images_dir = os.path.join(data_dir, 'images/er')
classification_dir = os.path.join(data_dir, 'classification')

beta1 = .15

normalizer_file = os.path.join(data_dir, 'normalizer')
if os.path.isfile(normalizer_file):
    normalizer = Normalizer.load(normalizer_file)
else:
    normalizer = Normalizer()
    normalizer.get_ref(ref_dir=er_images_dir)
    normalizer.save(normalizer_file)

avg_its_list = []
stain_ref = normalizer.stain_ref
file_list = glob(os.path.join(er_images_dir, '*'))
id_list = []
for image_file in tqdm(file_list):
    # store subject ID
    id = os.path.basename(image_file).split('_')[0]
    id_list.append(id)
    # get average intensities
    img = cp.array(Image.open(image_file))
    img = img.reshape((-1, 3))
    od = rgb2od(img)
    od = od[(cp.sum(od**2, axis=1) > beta1**2)]
    its = get_intensity(stain_ref, od)
    if its is None:
        avg_its = np.array([0, 0]) # for images with little tissue
    else:
        avg_its = cp.mean(its, axis=1)
        avg_its = cp.asnumpy(avg_its)
    avg_its_list.append(avg_its)

# save as csv file
avg_its_list = np.array(avg_its_list)
df = pd.DataFrame(avg_its_list, index=id_list, columns = ['blue', 'brown'])
df.to_csv(os.path.join(classification_dir, 'avg_intensities.csv'))
