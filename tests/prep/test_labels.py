import numpy as np
import pandas as pd
import pytest

import prep.labels
from prep.label_converter import DONT_KNOW, NO, YES

SEA_STARS = 92
BIOTA = 2
ECHINODERMS = 88
PEBBLE_GRAVEL = 245
BIOLOGENIC = 248
CORAL_RUBBLE = 249

SMALL_SHAPE = (8, 10)


@pytest.fixture
def points():
    """Two annotated points in a small synthetic image."""
    return pd.DataFrame(
        {
            "image_name": ["fake_image", "fake_image"],
            "row": [2, 5],
            "col": [3, 7],
            "label": [SEA_STARS, PEBBLE_GRAVEL],
        }
    )


class TestLoadPoints:
    def test_clips_points_past_the_edge(self, tmp_path):
        """A few published points sit one pixel outside the image."""
        csv = tmp_path / "split.csv"
        pd.DataFrame(
            {"image_name": ["a", "a"], "row": [1024, 3], "col": [7, 1360], "label": [SEA_STARS, SEA_STARS]}
        ).to_csv(csv, index=False)

        loaded = prep.labels.load_points(csv)

        assert loaded.row.tolist() == [1023, 3]
        assert loaded.col.tolist() == [7, 1359]

    def test_leaves_in_bounds_points_alone(self, tmp_path):
        csv = tmp_path / "split.csv"
        pd.DataFrame({"image_name": ["a"], "row": [10], "col": [20], "label": [SEA_STARS]}).to_csv(csv, index=False)

        loaded = prep.labels.load_points(csv)

        assert loaded.row.tolist() == [10]
        assert loaded.col.tolist() == [20]

    def test_reads_a_real_split(self, data_split_dir):
        loaded = prep.labels.load_points(data_split_dir / "public_labels_test.csv")

        assert list(loaded.columns) == ["image_name", "row", "col", "label"]
        assert loaded.row.between(0, prep.labels.IM_SHAPE[0] - 1).all()
        assert loaded.col.between(0, prep.labels.IM_SHAPE[1] - 1).all()


class TestBitVectorMatrix:
    def test_is_square_over_classes(self, df_id_lookups):
        matrix = prep.labels.bit_vector_matrix(df_id_lookups)
        assert matrix.shape == (len(df_id_lookups), len(df_id_lookups))

    def test_rows_match_the_per_class_vectors(self, df_id_lookups):
        matrix = prep.labels.bit_vector_matrix(df_id_lookups)
        position = df_id_lookups.index.get_loc(PEBBLE_GRAVEL)
        assert np.array_equal(matrix[position], df_id_lookups.loc[PEBBLE_GRAVEL, "bit_vector"])


class TestPointsToBitVectors:
    def test_one_row_per_point(self, points, df_id_lookups):
        vectors = prep.labels.points_to_bit_vectors(points, df_id_lookups)
        assert vectors.shape == (len(points), len(df_id_lookups))

    def test_rows_are_the_classes_bit_vectors(self, points, df_id_lookups):
        vectors = prep.labels.points_to_bit_vectors(points, df_id_lookups)
        assert np.array_equal(vectors[0], df_id_lookups.loc[SEA_STARS, "bit_vector"])
        assert np.array_equal(vectors[1], df_id_lookups.loc[PEBBLE_GRAVEL, "bit_vector"])

    def test_unknown_label_id_raises(self, df_id_lookups):
        unknown = pd.DataFrame({"row": [0], "col": [0], "label": [999999]})
        with pytest.raises(KeyError, match="999999"):
            prep.labels.points_to_bit_vectors(unknown, df_id_lookups)


class TestPointsToMask:
    def test_shape_is_classes_by_image(self, points, df_id_lookups):
        mask = prep.labels.points_to_mask(points, df_id_lookups, SMALL_SHAPE)
        assert mask.shape == (len(df_id_lookups), *SMALL_SHAPE)
        assert mask.dtype == np.int8

    def test_unannotated_pixels_are_dont_know(self, points, df_id_lookups):
        """Sparse points say nothing about the rest of the image."""
        mask = prep.labels.points_to_mask(points, df_id_lookups, SMALL_SHAPE)
        assert (mask[:, 0, 0] == DONT_KNOW).all()
        assert (mask != DONT_KNOW).any(axis=0).sum() == len(points)

    def test_annotated_pixel_carries_the_bit_vector(self, points, df_id_lookups):
        mask = prep.labels.points_to_mask(points, df_id_lookups, SMALL_SHAPE)
        assert np.array_equal(mask[:, 2, 3], df_id_lookups.loc[SEA_STARS, "bit_vector"])
        assert np.array_equal(mask[:, 5, 7], df_id_lookups.loc[PEBBLE_GRAVEL, "bit_vector"])

    def test_hierarchy_is_expanded_at_the_point(self, points, df_id_lookups):
        mask = prep.labels.points_to_mask(points, df_id_lookups, SMALL_SHAPE)
        position = df_id_lookups.index.get_loc

        # The sea stars point: self and ancestors positive, unrelated classes negative.
        assert mask[position(SEA_STARS), 2, 3] == YES
        assert mask[position(ECHINODERMS), 2, 3] == YES
        assert mask[position(BIOTA), 2, 3] == YES
        assert mask[position(PEBBLE_GRAVEL), 2, 3] == NO

        # The pebble/gravel point is an internal node, so its subtree stays unknown.
        assert mask[position(PEBBLE_GRAVEL), 5, 7] == YES
        assert mask[position(BIOLOGENIC), 5, 7] == DONT_KNOW
        assert mask[position(CORAL_RUBBLE), 5, 7] == DONT_KNOW

    def test_unknown_label_id_raises(self, df_id_lookups):
        unknown = pd.DataFrame({"row": [0], "col": [0], "label": [999999]})
        with pytest.raises(KeyError, match="999999"):
            prep.labels.points_to_mask(unknown, df_id_lookups, SMALL_SHAPE)


class TestParquetRoundTrip:
    def test_points_survive_a_write_and_read(self, points, tmp_path):
        """Storage is plain pandas; this guards the format, not our own codec."""
        path = tmp_path / "points.parquet"
        points.to_parquet(path, index=False)

        assert pd.read_parquet(path).equals(points)

    def test_masks_rebuilt_from_parquet_are_identical(self, points, df_id_lookups, tmp_path):
        path = tmp_path / "points.parquet"
        points.to_parquet(path, index=False)

        from_memory = prep.labels.points_to_mask(points, df_id_lookups, SMALL_SHAPE)
        from_disk = prep.labels.points_to_mask(pd.read_parquet(path), df_id_lookups, SMALL_SHAPE)

        assert np.array_equal(from_memory, from_disk)
