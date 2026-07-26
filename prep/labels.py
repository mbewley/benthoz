"""Sparse point labels for BENTHOZ-2015.

BENTHOZ annotations are sparse points - roughly 40 per image - not dense masks.
An earlier version of this pipeline rasterised them to one
(n_classes, height, width) int8 array per image and stored that as .npz. It cost
206 MB of memory per image to carry about 50 meaningful values, and 263 MB on
disk for the training split.

The points are now kept sparse - one parquet file per split, about 1 MB for the
whole training set - and rasterised on demand, only where a model actually wants
a dense mask. points_to_mask reproduces the old .npz arrays exactly.

Writing a split is just pandas::

    load_points("data_splits/public_labels_train.csv").to_parquet(path)
"""

import numpy as np
import pandas as pd

from prep.label_converter import DONT_KNOW

IM_SHAPE = (1024, 1360)


def load_points(csv_path, im_shape=IM_SHAPE):
    """
    Read a public_labels_*.csv split, clipping points that fall outside the image.

    A handful of points in the published splits sit one pixel past the edge
    (row 1024, col 1360); they are clipped back to the last valid pixel.

    :param csv_path: Path to a public_labels_{train,cal,test}.csv.
    :param im_shape: (rows, cols) of the images the points refer to.
    :return: A dataframe of image_name, row, col, label.
    """
    points = pd.read_csv(csv_path)
    points["row"] = points.row.clip(0, im_shape[0] - 1)
    points["col"] = points.col.clip(0, im_shape[1] - 1)
    return points


def bit_vector_matrix(df_id_lookups):
    """
    Stack the per-class bit vectors into one array.

    :param df_id_lookups: A dataframe from build_hierarchy_from_id_lookup.
    :return: An (n_classes, n_classes) array whose row i is the bit vector of
            the class at position i.
    """
    return np.stack(df_id_lookups.bit_vector.values)


def points_to_bit_vectors(points, df_id_lookups):
    """
    Look up the multilabel bit vector for each point.

    This is what a patch classifier wants: one target vector per annotated
    point, without ever building a dense mask.

    :param points: A dataframe with a label column.
    :param df_id_lookups: A dataframe from build_hierarchy_from_id_lookup.
    :return: An (n_points, n_classes) int8 array.
    """
    return bit_vector_matrix(df_id_lookups)[_class_positions(points, df_id_lookups)]


def points_to_mask(points, df_id_lookups, im_shape=IM_SHAPE):
    """
    Rasterise a single image's points into a dense multilabel mask.

    Every pixel starts as DONT_KNOW. Each annotated point is set to its class
    bit vector, which marks that class and its ancestors YES, its descendants
    DONT_KNOW, and every other class NO.

    Coordinates are assumed to already sit inside im_shape; load_points does that
    clipping, and this does not repeat it.

    :param points: A dataframe of row, col, label for one image.
    :param df_id_lookups: A dataframe from build_hierarchy_from_id_lookup.
    :param im_shape: (rows, cols) of the image.
    :return: An (n_classes, rows, cols) int8 array.
    """
    bit_vectors = points_to_bit_vectors(points, df_id_lookups)

    mask = np.full((len(df_id_lookups), *im_shape), DONT_KNOW, dtype=np.int8)
    mask[:, points.row.values, points.col.values] = bit_vectors.T
    return mask


def _class_positions(points, df_id_lookups):
    """
    Map label ids to their row position in the id lookup table.

    :param points: A dataframe with a label column.
    :param df_id_lookups: A dataframe from build_hierarchy_from_id_lookup.
    :return: An array of positional indices, one per point.
    :raises KeyError: If any label id is absent from the lookup table.
    """
    positions = df_id_lookups.index.get_indexer(points.label.values)
    if (positions < 0).any():
        missing = sorted(set(np.asarray(points.label.values)[positions < 0].tolist()))
        raise KeyError("Label ids missing from the id lookup table: {}".format(missing))
    return positions
