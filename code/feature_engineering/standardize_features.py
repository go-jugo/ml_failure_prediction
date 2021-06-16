import pandas as pd
from sklearn.preprocessing import StandardScaler
from monitoring.time_it import timing

@timing
def standardize_features(df, errorcode_col, scaler = StandardScaler()):
    """
    Standardize features for machine learning application, e.g. z-transformation.

    :param df: pandas dataframe
    :param errorcode_col: target errorcode column in dask dataframe (String)
    :param scaler: strategy to standardize features, e.g. "StandardScaler" = z-transformation
    :return: pandas dataframe
    """
    feature_columns = [col for col in df.columns if col not in ['global_timestamp', errorcode_col, 'samples_used', 'window_start', 'window_end', 'window_length']]
    df[feature_columns] = df[feature_columns].to_numpy()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df