import pandas as pd
import os
import urllib
import logging

S3_ROOT = 'https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/AUV/'


def web_location_to_url(web_location):
    im_path_root, im_filename = web_location.split('images')
    im_filename = im_filename.replace('png', 'tif').strip('/')
    timestamp = '_'.join(im_path_root.split('/')[1].split('_')[:2])[1:]
    im_subfolder = 'i' + timestamp + '_gtif'
    im_path_full = im_path_root + im_subfolder + '/' + im_filename
    url = S3_ROOT + im_path_full
    return url


def fetch_image(url, image_folder):
    im_filename = os.path.split(url)[-1]
    im_path = os.path.join(image_folder, im_filename)
    if not os.path.exists(im_path):
        try:
            urllib.request.urlretrieve(url, im_path)
        except urllib.error.HTTPError:
            pass
    else:
        logging.warning('Skipping download of existing image: {}'.format(im_path))
    return im_path


def fetch_all_images(image_list_file, image_destination_folder):
    df_images = (pd
                 .read_csv(image_list_file)
                 .query('image__id != "image__id"')  # Filter out 8 junk rows
                 )
    df_images.head()
    for row in df_images.itertuples():
        url = web_location_to_url(row.web_location)
        fetch_image(url, image_destination_folder)


if __name__ == '__main__':
    fetch_all_images()
