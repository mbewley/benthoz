import pandas as pd
import pytest

from prep.label_converter import DONT_KNOW, NO, YES

# From testing_data/idlookups_test.csv.
SEA_STARS = 92  # Biota: Echinoderms: Sea stars, a leaf
ECHINODERMS = 88
BIOTA = 2
PEBBLE_GRAVEL = 245  # Physical: Substrate: Unconsolidated (soft): Pebble / gravel
BIOLOGENIC = 248
CORAL_RUBBLE = 249
SUBSTRATE = 239
UNCONSOLIDATED = 240
PHYSICAL = 238


@pytest.fixture
def df_id_lookups_raw(id_lookup_file):
    return pd.read_csv(id_lookup_file, index_col=0)


class TestBuildHierarchy:
    def test_preserves_every_class(self, df_id_lookups, df_id_lookups_raw):
        assert len(df_id_lookups) == len(df_id_lookups_raw)

    def test_biota_is_prepended_to_its_children(self, df_id_lookups):
        """Biota and Physical are not part of their children's names in the source file."""
        assert df_id_lookups.loc[SEA_STARS, "parsed_name"] == ["Biota", "Echinoderms", "Sea stars"]

    def test_physical_is_prepended_to_its_children(self, df_id_lookups):
        assert df_id_lookups.loc[SUBSTRATE, "parsed_name"] == ["Physical", "Substrate"]

    def test_depth_counts_the_prepended_root(self, df_id_lookups):
        assert df_id_lookups.loc[BIOTA, "depth"] == 1
        assert df_id_lookups.loc[ECHINODERMS, "depth"] == 2
        assert df_id_lookups.loc[SEA_STARS, "depth"] == 3
        assert df_id_lookups.loc[CORAL_RUBBLE, "depth"] == 6

    def test_child_name_is_the_last_segment(self, df_id_lookups):
        assert df_id_lookups.loc[SEA_STARS, "child_name"] == "Sea stars"


class TestAncestorsAndDescendants:
    def test_ancestors_are_ordered_root_first(self, df_id_lookups):
        assert list(df_id_lookups.loc[CORAL_RUBBLE, "ancestor_id_list"]) == [
            PHYSICAL,
            SUBSTRATE,
            UNCONSOLIDATED,
            PEBBLE_GRAVEL,
            BIOLOGENIC,
        ]

    def test_root_has_no_ancestors(self, df_id_lookups):
        assert list(df_id_lookups.loc[BIOTA, "ancestor_id_list"]) == []

    def test_leaf_has_no_descendants(self, df_id_lookups):
        assert list(df_id_lookups.loc[SEA_STARS, "descendant_id_list"]) == []

    def test_descendants_include_the_whole_subtree(self, df_id_lookups):
        assert set(df_id_lookups.loc[PEBBLE_GRAVEL, "descendant_id_list"]) == {BIOLOGENIC, CORAL_RUBBLE}

    def test_the_two_roots_do_not_mix(self, df_id_lookups):
        biota_subtree = set(df_id_lookups.loc[BIOTA, "descendant_id_list"])
        physical_subtree = set(df_id_lookups.loc[PHYSICAL, "descendant_id_list"])
        assert biota_subtree.isdisjoint(physical_subtree)
        assert SEA_STARS in biota_subtree
        assert PEBBLE_GRAVEL in physical_subtree


class TestBitVectors:
    def test_length_matches_class_count(self, df_id_lookups):
        assert len(df_id_lookups.loc[SEA_STARS, "bit_vector"]) == len(df_id_lookups)

    def test_self_and_ancestors_are_yes(self, df_id_lookups):
        bit_vector = df_id_lookups.loc[PEBBLE_GRAVEL, "bit_vector"]
        position = df_id_lookups.index.get_loc

        assert bit_vector[position(PEBBLE_GRAVEL)] == YES
        for ancestor in (PHYSICAL, SUBSTRATE, UNCONSOLIDATED):
            assert bit_vector[position(ancestor)] == YES

    def test_descendants_are_dont_know(self, df_id_lookups):
        """A point labelled at an internal node says nothing about its subtree."""
        bit_vector = df_id_lookups.loc[PEBBLE_GRAVEL, "bit_vector"]
        position = df_id_lookups.index.get_loc

        for descendant in (BIOLOGENIC, CORAL_RUBBLE):
            assert bit_vector[position(descendant)] == DONT_KNOW

    def test_unrelated_classes_are_no(self, df_id_lookups):
        bit_vector = df_id_lookups.loc[PEBBLE_GRAVEL, "bit_vector"]
        position = df_id_lookups.index.get_loc

        assert bit_vector[position(BIOTA)] == NO
        assert bit_vector[position(SEA_STARS)] == NO

    def test_leaf_has_no_dont_know_entries(self, df_id_lookups):
        bit_vector = df_id_lookups.loc[SEA_STARS, "bit_vector"]
        assert DONT_KNOW not in bit_vector
