import pandas as pd
import dask.dataframe as dd
from monitoring.time_it import timing


@timing
def remove_error_codes(df, dependent_variable = 'components.cont.conditions.logic.errorCode', skip=True):
    """
    Remove errorcode columns from features except the target errorcode column. Usually, this is not necessary.

    :param df: dask dataframe
    :param dependent_variable: target errorcode column (String)
    :param skip: skip function (Binary)
    :return: dask dataframe
    """
    if not skip:
        errorCode_columns = [col for col in df.columns if 'errorCode' in col]
        errorCode_columns.remove(dependent_variable)
        df = df.drop(columns=errorCode_columns)
    return df