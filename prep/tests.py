import os
import logging

import cv2
import pandas as pd

from patches import get_image, get_patches_from_image, write_patches_as_images
#from benthoz.prep.patches import get_image, get_patches_from_image, write_patches_as_images

DATA_SPLIT_DIR = '../data_splits'
IMAGES_DIR = '../data/benthoz-2015'
OUT_DIR = '../data/output_patches'

logging.root.setLevel(logging.INFO)
images_list = []

for data_split_file in os.listdir(DATA_SPLIT_DIR):
    data_split = pd.read_csv(os.path.join(DATA_SPLIT_DIR, data_split_file))
    images_list.extend(data_split.image_name.unique())
logging.info('Data splits refers to %d unique images' % len(images_list))
print(images_list[0])
images_found = [os.path.splitext(p)[0] for p in os.listdir(IMAGES_DIR)]
logging.info('Image files found locally: %d' % len(images_found))

training_points = pd.read_csv(os.path.join(DATA_SPLIT_DIR, 'public_labels_train.csv'))

for image_name, image_points in training_points.groupby('image_name'):
    try:
        im = get_image(os.path.join(IMAGES_DIR, image_name+'.png'))
    except IOError as e:
        logging.debug("Couldn't find image {}".format(image_name))
    else:
        image_points = image_points.set_index(['row', 'col'])
        patches = get_patches_from_image(im, image_points.index, 127)
        write_patches_as_images(image_name, patches=patches, labels=image_points.label, out_dir=OUT_DIR)

