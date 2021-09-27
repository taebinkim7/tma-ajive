##### activate conda env for stain normalization first
import os
import numpy as np
import cupy as cp
import pandas as pd

from glob import glob
from PIL import Image
from stain_norm import Normalizer
from stain_norm.utils import rgb2od, get_intensity


data_dir = '/datastore/nextgenout5/share/labs/smarronlab/tkim/data/tma_9830'
er_images_dir = os.path.join(data_dir, 'images/er')
classification_dir = os.path.join(data_dir, 'classification')

beta1 = .15

normalizer_file = os.path.join(data_dir, 'normalizer')
if os.path.isfile(normalizer_file):
    normalizer = Normalizer.load(normalizer_file)
else:
    normalizer = Normalizer()
    normalizer.get_ref(ref_dir=input_dir)
    normalizer.save(normalizer_file)

avg_its_list = []
stain_ref = normalizer.stain_ref
file_list = glob(er_images_dir)
index_list = []
for image_file in file_list:
    index = os.path.basename(image_file).split('_')[0]
    index_list.append(index)
    img = cp.array(Image.open(image_file))
    od = rgb2od(img)
    od = od[(cp.sum(od**2, axis=1) > beta1**2)]
    its = get_intensity(stain_ref, od.T)
    if its is None:
        avg_its = np.array([0, 0])
    else:
        avg_its = cp.mean(its, axis=1)
        avg_its = cp.asnumpy(avg_its)
    avg_its_list.append(avg_its)

df = pd.DataFrame(avg_its, index=index_list, columns = ['blue', 'brown'])
df.to_csv(os.path.join(classification_dir, 'avg_intensities.csv'))
