import os
import numpy as np

from argparse import ArgumentParser
from glob import glob
from PIL import Image


parser = ArgumentParser()
parser.add_argument('--input_dir', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
args = parser.parse_args()

input_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.input_dir, 'images')
output_dir = os.path.join('/datastore/lbcfs/labs/smarronlab/tkim/data', args.output_dir, 'images')

def transfer_dir(image_type):
    file_list = glob(os.path.join(input_dir, image_type, '*'))
    for image_file in file_list:
        file_name = os.path.basename(image_file)
        img = np.array(Image.open(image_file))
        save_dir = os.path.join(output_dir, image_type)
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, file_name)
        Image.fromarray(img).save(save_file)

transfer_dir('he')
transfer_dir('er')
