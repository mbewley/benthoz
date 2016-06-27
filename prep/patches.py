import os
import cv2
import pandas as pd

def get_image(image_path):
    if os.path.exists(image_path):
        im = cv2.imread(image_path)
        return im
    else:
        raise IOError("Couldn't find image {}".format(image_path))

def get_patches_from_image(im, coords_index, patch_size, return_patches=True):
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

    patches = pd.DataFrame(index=coords_index)
    patches['min_r'] = r - patch_radius
    patches['max_r'] = r + patch_radius
    patches['min_c'] = c - patch_radius
    patches['max_c'] = c + patch_radius

    patches['not_cropped'] = (patches.min_r >= 0) & (patches.max_r < im.shape[0]) & \
                             (patches.min_c >= 0) & (patches.max_c < im.shape[1])
    patches.loc[patches.min_r < 0, 'min_r'] = 0
    patches.loc[patches.min_c < 0, 'min_c'] = 0
    image_patches = {}

    if not return_patches:
        return patches.index
    else:
        for (r, c), p in patches.iterrows():
            image_patches[(r,c)] = im[p.min_r:p.max_r + 1, p.min_c:p.max_c + 1]
        return patches, image_patches

def write_patches_as_images(image_patches, group, out_dir):
    """

    :param image_patches: Dictionary of key (r,c), value (patch RGB numpy array)
    :param group: Group including label dictionary of key (r,c), value (class labels), image_name and patch row and column
    :param out_dir: Directory to write images
    :return:
    """
    for r, c in image_patches:
        label = group.label[(r,c)]
        image_path = os.path.join(out_dir, '{}'.format(label))
        if not os.path.exists(image_path):
            os.mkdir(image_path)
        cv2.imwrite(os.path.join(image_path, '{}.png'.format(group.image_name[(r,c)]+'_'+str(r)+'_'+str(c))), image_patches[(r,c)])
