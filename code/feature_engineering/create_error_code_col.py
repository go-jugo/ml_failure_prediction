from tools.series_list_to_df import series_list_to_df

def create_error_code_col(df, error_code_col):
    """
    Save errorcode column in separate dask dataframe

    :param df: dask dataframe
    :param error_code_col: errorcode column of dask dataframe
    :return: dask dataframe
    """

    df = df[[error_code_col]]
    return df