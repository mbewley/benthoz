import pandas as pd
import numpy as np

DONT_KNOW = -1
NO = 0
YES = 1

# Now for a more complicate trick - convert "parsed_name" to a list of node IDs, instead of strings

def get_ancestor_ids(label_id, df_id_lookups):
    """

    :param label_id: The id of the leaf node.
    :param df_id_lookups: A dataframe read from idlookups.csv.
    :return: The list of ancestors, not including the provided label ID.
    """
    parsed_name = df_id_lookups.loc[label_id, 'parsed_name']
    ancestor_parsed_ids = []
    for i in range(len(parsed_name)-1): # -1 to exclude the current class from the list.
        ancestor_parsed_name = parsed_name[:i + 1]
        ancestor_parsed_id = df_id_lookups[df_id_lookups.parsed_name.apply(lambda d: d == ancestor_parsed_name)].index[
            0]
        ancestor_parsed_ids.append(ancestor_parsed_id)
    return ancestor_parsed_ids


def get_descendant_ids(label_id, df_id_lookups):
    """

    :param label_id: The id of the leaf node.
    :param df_id_lookups: A dataframe read from idlookups.csv.
    :return: The list of descendants, not including the provided label ID.
    """
    # Get descendents NOT including this label_id.
    descendant_idx = df_id_lookups.ancestor_id_list.apply(lambda d: label_id in d)
    descendant_label_ids = df_id_lookups.loc[descendant_idx].index.values
    return descendant_label_ids


def get_bit_vector(label_id, df_id_lookups):
    """

    :param label_id: The explicitly provided class label.
    :param df_id_lookups: A dataframe read from idlookups.csv.
    :return: The bit vector for a particular class.
    """
    k = len(df_id_lookups)
    bit_vector = np.zeros(k, dtype=np.int8)
    bit_vector[:] = NO

    idx = np.where(df_id_lookups.index == label_id)[0]
    bit_vector[idx] = YES

    for ancestor_id in df_id_lookups.loc[label_id, 'ancestor_id_list']:
        idx = np.where(df_id_lookups.index == ancestor_id)[0]
        bit_vector[idx] = YES

    for descendant_id in df_id_lookups.loc[label_id, 'descendant_id_list']:
        idx = np.where(df_id_lookups.index == descendant_id)[0]
        bit_vector[idx] = DONT_KNOW

    return bit_vector

def build_hierarchy_from_id_lookup(id_lookup_file='idlookups.csv'):
    """
    Infer the hierarchical structure of the Catami Hierarchy from the provided idlookups.csv.
    :param id_lookup_file: The idlookups.csv file attached to the Scientific Data paper.
    :return:
    """
    df_id_lookups = pd.read_csv(id_lookup_file, index_col=0)

    # The naming convention separates layers of the hierarchy with a colon ':', so we can break this into a list of descendents, and calculate the depth of the tree.
    df_id_lookups['parsed_name'] = df_id_lookups.name.apply(lambda s: s.split(': '))
    df_id_lookups['depth'] = df_id_lookups.parsed_name.apply(lambda d: len(d))

    # The two top nodes "Biota" and "Physical" are not prepended to their children, so we need to do this manually.
    # Manually define biota and physical children
    biota_kids = ['Worms', 'Sponges', 'Seagrasses', 'Molluscs', 'Macroalgae', 'Jellies', 'Fishes', 'Echinoderms',
                  'Crustacea',
                  'Cnidaria', 'Bryozoa', 'Bioturbation', 'Bacterial mats', 'Ascidians']

    physical_kids = ['Substrate']

    # Prepend them to name lists, and add to depth.
    biota_inds = df_id_lookups.parsed_name.apply(lambda d: d[0] in biota_kids)
    df_id_lookups.loc[biota_inds, 'depth'] += 1
    df_id_lookups.loc[biota_inds, 'parsed_name'] = df_id_lookups.loc[biota_inds, 'parsed_name'].apply(
        lambda d: ['Biota'] + d)

    physical_inds = df_id_lookups.parsed_name.apply(lambda d: d[0] in physical_kids)
    df_id_lookups.loc[physical_inds, 'depth'] += 1
    df_id_lookups.loc[physical_inds, 'parsed_name'] = df_id_lookups.loc[physical_inds, 'parsed_name'].apply(
        lambda d: ['Physical'] + d)

    # Create columns for ancestor and descendant lists.
    df_id_lookups['child_name'] = df_id_lookups.parsed_name.apply(lambda d: d[-1])

    df_id_lookups['ancestor_id_list'] = [get_ancestor_ids(d, df_id_lookups) for d in df_id_lookups.index]

    df_id_lookups['descendant_id_list'] = [get_descendant_ids(d, df_id_lookups) for d in df_id_lookups.index]

    # Create a multilabel, one hot encoded bit vector for each class, taking into account the hierarchy of ancestors, and unspecified descendants.
    # We now want to represent this class hierarchy as a bit-vector. Each class index has a unique bit in the vector. A root level class will turn on a single bit. A depth 4 class will turn on 4 bits.
    df_id_lookups['bit_vector'] = [get_bit_vector(d, df_id_lookups) for d in df_id_lookups.index]
    df_id_lookups

    return df_id_lookups


