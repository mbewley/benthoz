import os
import cv2
import pandas as pd


def get_image(image_path):
    if os.path.exists(image_path):
        im = cv2.imread(image_path)
        return im
    else:
        raise IOError("Couldn't find image {}".format(image_path))


def get_patches_from_image(im, coords_index, patch_size):
    """


    :param withempty: Whether to include 'na' patch entries in the index.
    :param im: An image as a numpy array. Patch will be taken from all channels, as (rows, columns, channels)
    :param coords_index: Index of (row, col) points to use as patch centres
    :param patch_size: Size of patches in pixels (e.g. 15 = square 15x15 patches)
    :param return_patches: Whether to return the actual patches, or just the index of valid patches
    :return: An index of valid patches (if return_patches), or a pandas series of numpy array patches, indexed by
            coords_index.
    """
    assert (patch_size % 2 == 1) and (patch_size > 0)
    patch_radius = patch_size / 2

    r = coords_index.get_level_values('row').values
    c = coords_index.get_level_values('col').values

    patches_bounds = pd.DataFrame(index=coords_index)
    patches_bounds['min_r'] = r - patch_radius
    patches_bounds['max_r'] = r + patch_radius
    patches_bounds['min_c'] = c - patch_radius
    patches_bounds['max_c'] = c + patch_radius

    patches_bounds['not_cropped'] = (patches_bounds.min_r >= 0) & (patches_bounds.max_r < im.shape[0]) & \
                                    (patches_bounds.min_c >= 0) & (patches_bounds.max_c < im.shape[1])
    patches_bounds.loc[patches_bounds.min_r < 0, 'min_r'] = 0
    patches_bounds.loc[patches_bounds.min_c < 0, 'min_c'] = 0
    patches = []
    for (r, c), p in patches_bounds.iterrows():
        patches.append(im[p.min_r:p.max_r + 1, p.min_c:p.max_c + 1])
    patches = pd.Series(patches, index=coords_index, name='patch')
    return patches


def write_patches_as_images(image_name, patches, labels, out_dir):
    """
    Takes a set of patches from a single image, and writes them to disk.

    :param patches: pandas.Series of index (r,c), value (patch RGB numpy array)
    :param image_points: pandas.Series of (r,c), value (class labels)
    :param out_dir: Base directory to write images
    """
    df = pd.concat([patches, labels], axis=1)

    for (r, c), dfrow in df.iterrows():
        image_path = os.path.join(out_dir, '{}'.format(dfrow.label))
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        file_name = '{}_{}_{}.png'.format(image_name, r, c)
        cv2.imwrite(os.path.join(image_path, file_name), df.patch[(r, c)])
