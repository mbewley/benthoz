"""Shared pytest fixtures.

Living at the repo root, this file also puts the root on sys.path, so ``import
prep.*`` resolves no matter which directory pytest is invoked from. Test data
paths are anchored to this file rather than the working directory for the same
reason.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent
TESTING_DATA_DIR = REPO_ROOT / "testing_data"
DATA_SPLIT_DIR = REPO_ROOT / "data_splits"

TEST_IMAGE_NAME = "PR_20081006_232302_383_LC16"


@pytest.fixture(scope="session")
def repo_root():
    """The repository root, for tests that read checked-in data files."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def testing_data_dir():
    """The directory holding small checked-in test inputs."""
    return TESTING_DATA_DIR


@pytest.fixture(scope="session")
def data_split_dir():
    """The directory holding the published train/cal/test splits."""
    return DATA_SPLIT_DIR


@pytest.fixture(scope="session")
def id_lookup_file(testing_data_dir):
    """A cut-down idlookups.csv covering both roots and depths 1 to 6."""
    return testing_data_dir / "idlookups_test.csv"


@pytest.fixture(scope="session")
def df_id_lookups(id_lookup_file):
    """The class hierarchy built from the cut-down id lookup table."""
    from prep.label_converter import build_hierarchy_from_id_lookup

    return build_hierarchy_from_id_lookup(id_lookup_file)


@pytest.fixture(scope="session")
def test_image_path(testing_data_dir):
    """Path to the single checked-in sample image."""
    return testing_data_dir / (TEST_IMAGE_NAME + ".png")
