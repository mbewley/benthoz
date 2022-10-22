import pandas as pd
import os
import logging
import requests
from urllib3 import Retry
from requests.adapters import HTTPAdapter

import tqdm

S3_ROOT = "https://s3-ap-southeast-2.amazonaws.com/imos-data/IMOS/AUV/"


def web_location_to_url(web_location):
    im_path_root, im_filename = web_location.split("images")
    im_filename = im_filename.replace("png", "tif").strip("/")
    timestamp = "_".join(im_path_root.split("/")[1].split("_")[:2])[1:]
    im_subfolder = "i" + timestamp + "_gtif"
    im_path_full = im_path_root + im_subfolder + "/" + im_filename
    return S3_ROOT + im_path_full


def fetch_image(url, image_folder, session):
    im_filename = os.path.split(url)[-1]
    im_path = os.path.join(image_folder, im_filename)
    if not os.path.exists(im_path):
        response = session.get(url)
        if response.ok:
            im = response.content
            with open(im_path, "wb") as file:
                file.write(im)
        else:
            logging.error(f"Failed to fetch image {url}")
    else:
        logging.warning(f"Skipping download of existing image: {im_path}")
    return im_path


def fetch_all_images(image_list_file, image_destination_folder):
    df_images = pd.read_csv(image_list_file).query('image__id != "image__id"')  # Filter out 8 junk rows
    df_images.head()

    with requests.Session() as s:
        retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[408, 500])
        s.mount("https://", HTTPAdapter(max_retries=retries))

        for row in tqdm.tqdm(df_images.itertuples(), total=len(df_images)):
            url = web_location_to_url(row.web_location)

            fetch_image(url, image_destination_folder, session=s)


if __name__ == "__main__":
    fetch_all_images()
