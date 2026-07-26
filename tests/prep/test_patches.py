import numpy as np
import pandas as pd
import pytest

import prep.patches

PATCH_SIZE = 127

# Points chosen to cover the interior, the left edge, and the bottom edge.
P_INTERIOR = (119, 647)
P_LEFT_EDGE = (155, 50)
P_INTERIOR_2 = (942, 892)
P_BOTTOM_EDGE = (989, 365)
ALL_POINTS = [P_INTERIOR, P_LEFT_EDGE, P_INTERIOR_2, P_BOTTOM_EDGE]


@pytest.fixture
def im(test_image_path):
    return prep.patches.get_image(str(test_image_path))


@pytest.fixture
def coords_index():
    return pd.MultiIndex.from_tuples(ALL_POINTS, names=["row", "col"])


class TestGetImage:
    def test_shape_and_dtype(self, im):
        assert im.shape == (1024, 1360, 3)
        assert im.dtype == np.uint8

    def test_missing_image_raises(self, tmp_path):
        with pytest.raises(IOError):
            prep.patches.get_image(str(tmp_path / "does_not_exist.png"))


class TestGetPatchesFromImage:
    def test_patch_count(self, im, coords_index):
        patches = prep.patches.get_patches_from_image(im, coords_index, PATCH_SIZE)
        assert len(patches) == len(coords_index)

    def test_patch_shapes(self, im, coords_index):
        """Interior patches are square; edge patches are cropped to the image."""
        patches = prep.patches.get_patches_from_image(im, coords_index, PATCH_SIZE)
        assert patches[P_INTERIOR].shape == (127, 127, 3)
        assert patches[P_LEFT_EDGE].shape == (127, 64 + 50, 3)
        assert patches[P_INTERIOR_2].shape == (127, 127, 3)
        assert patches[P_BOTTOM_EDGE].shape == (63 + 1024 - 989, 127, 3)

    def test_patch_values(self, im, coords_index):
        patches = prep.patches.get_patches_from_image(im, coords_index, PATCH_SIZE)
        r, c = P_INTERIOR
        assert np.array_equal(patches[P_INTERIOR], im[r - 63 : r + 64, c - 63 : c + 64, :])
        r, c = P_LEFT_EDGE
        assert np.array_equal(patches[P_LEFT_EDGE], im[r - 63 : r + 64, : c + 64, :])
        r, c = P_INTERIOR_2
        assert np.array_equal(patches[P_INTERIOR_2], im[r - 63 : r + 64, c - 63 : c + 64, :])
        r, c = P_BOTTOM_EDGE
        assert np.array_equal(patches[P_BOTTOM_EDGE], im[r - 63 :, c - 63 : c + 64, :])

    @pytest.mark.parametrize(
        "point, expected_mean",
        [
            (P_INTERIOR, 49.5024490049),
            (P_LEFT_EDGE, 86.4618041166),
            (P_INTERIOR_2, 45.5314030628),
            (P_BOTTOM_EDGE, 72.5908725695),
        ],
    )
    def test_channel_means_unchanged(self, im, coords_index, point, expected_mean):
        """Regression guard on pixel content. Channel 0 is Blue, as opencv reads BGR."""
        patches = prep.patches.get_patches_from_image(im, coords_index, PATCH_SIZE)
        assert patches[point][:, :, 0].mean() == pytest.approx(expected_mean)

    def test_even_patch_size_rejected(self, im, coords_index):
        with pytest.raises(AssertionError):
            prep.patches.get_patches_from_image(im, coords_index, 128)

    def test_discard_cropped_keeps_only_full_patches(self, im, coords_index):
        patches = prep.patches.get_patches_from_image(im, coords_index, PATCH_SIZE, discard_cropped=True)
        assert len(patches) == 2
        assert set(patches.index) == {P_INTERIOR, P_INTERIOR_2}


class TestWritePatchesAsImages:
    def test_roundtrip_through_disk(self, im, coords_index, tmp_path):
        patches = prep.patches.get_patches_from_image(im, coords_index, PATCH_SIZE)
        labels = pd.Series([2, 238, 2, 238], index=coords_index, name="label")

        prep.patches.write_patches_as_images("test_image", patches, labels, str(tmp_path))

        for r, c in coords_index:
            written = prep.patches.get_image(str(tmp_path / str(labels[(r, c)]) / f"test_image_{r}_{c}.png"))
            assert np.array_equal(patches[(r, c)], written)

    def test_creates_missing_output_directory(self, im, coords_index, tmp_path):
        """The label subdirectory and any missing parents are created on demand."""
        patches = prep.patches.get_patches_from_image(im, coords_index, PATCH_SIZE)
        labels = pd.Series([2, 238, 2, 238], index=coords_index, name="label")
        out_dir = tmp_path / "does" / "not" / "exist"

        prep.patches.write_patches_as_images("test_image", patches, labels, str(out_dir))

        assert (out_dir / "2").is_dir()
