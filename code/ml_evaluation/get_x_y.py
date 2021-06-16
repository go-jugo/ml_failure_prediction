from monitoring.time_it import timing

@timing
def get_x_y(df, target_variables_column, errorcode):
    """
    Select features (independet variables) and target column (depended variable) for machine learning application

    :param df: pandas dataframe
    :param target_variables_column: target errorcode column in pandas dataframe (String)
    :param errorcode: errorcode (Integer)
    :return: tuple of pandas dataframe. First entry = features. Second entry = target column
    """
    y = df[target_variables_column]
    y[y != errorcode] = -1
    y = y.astype(int)
    X = df.drop(columns=[target_variables_column])
    return (X, y)