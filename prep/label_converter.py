import pandas as pd


# Now for a more complicate trick - convert "parsed_name" to a list of node IDs, instead of strings

def get_ancestor_ids(child_id, df_id_lookups):
    """

    :param child_id: The id of the leaf node.
    :param df_id_lookups: A dataframe read from idlookups.csv.
    :return:
    """
    parsed_name = df_id_lookups.loc[child_id, 'parsed_name']
    ancestor_parsed_ids = []
    for i in range(len(parsed_name)):
        ancestor_parsed_name = parsed_name[:i + 1]
        ancestor_parsed_id = df_id_lookups[df_id_lookups.parsed_name.apply(lambda d: d == ancestor_parsed_name)].index[
            0]
        ancestor_parsed_ids.append(ancestor_parsed_id)
    return ancestor_parsed_ids


def build_hierarchy_from_id_lookup(id_lookup_file='idlookups.csv'):
    """
    Infer the hierarchical structure of the Catami Hierarchy from the provided idlookups.csv.
    :param id_lookup_file: The idlookups.csv file attached to the Scientific Data paper.
    :return:
    """
    df_id_lookups = pd.read_csv(id_lookup_file, index_col=0)
    df_id_lookups['parsed_name'] = df_id_lookups.name.apply(lambda s: s.split(': '))
    df_id_lookups['depth'] = df_id_lookups.parsed_name.apply(lambda d: len(d))

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

    df_id_lookups['child_name'] = df_id_lookups.parsed_name.apply(lambda d: d[-1])
