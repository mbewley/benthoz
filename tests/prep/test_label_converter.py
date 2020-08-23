import pytest
import os

import pandas as pd

import prep.label_converter

TEST_INPUT_DIR = '../testing_data'
ID_LOOKUP_TEST_FILE = os.path.join(TEST_INPUT_DIR, 'idlookups_test.csv')

class TestLabelConverter:

    @pytest.fixture
    def df_id_lookups(self):
        df_id_lookups = prep.label_converter.build_hierarchy_from_id_lookup(ID_LOOKUP_TEST_FILE)
        return df_id_lookups

    @pytest.fixture
    def df_id_lookups_raw(self):
        df = pd.read_csv(ID_LOOKUP_TEST_FILE, index_col=0)
        return df


    def test_hierarchy_build(self, df_id_lookups, df_id_lookups_raw):
        assert len(df_id_lookups) == len(df_id_lookups_raw)
