from monitoring.time_it import timing

columns_to_remove = ['global_timestamp', 'samples_used', 'window_start', 'window_end', 'window_length']


@timing
def remove_global_timestamp(df):
    """
    remove non-feature columns and irrelevant feature columns

    :param df: pandas dataframe
    :return: pandas dataframe
    """
    df = df.drop(columns=columns_to_remove)
    list_to_remove = ['__sum_values', '__length']
    features_to_remove = []
    for element in list_to_remove:
        features = [col for col in df.columns if col.endswith(element)]
        features_to_remove.extend(features)
    df = df.drop(columns=features_to_remove)
    return df