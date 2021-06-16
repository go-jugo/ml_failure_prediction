import pandas as pd
import dask.dataframe as dd
from monitoring.time_it import timing
from tools.dask_repartition import dask_repartition

#function takes Dask- or Pandas-dataframe and converts categorial variables into dummy variables
@timing
def one_hot_encode_categories(df, errorcode_col, v_dask=True):
    """
    Convert categorical variables into dummy variables

    :param df: dask dataframe
    :param errorcode_col: errorcode column (String)
    :param v_dask: use dask dataframe format (Binary)
    :return: dask dataframe
    """

    non_numeric_columns_list = list(df.select_dtypes(exclude=['number', 'datetime']).columns)

    errorcode_column_list = [col for col in df.columns if 'errorCode' in col]
    errorcode_column_list.remove(errorcode_col)
    non_numeric_columns_list.extend(errorcode_column_list)

    df_non_numeric = df[non_numeric_columns_list].astype(str)
    if v_dask:
        df_non_numeric = df_non_numeric.categorize()
    if len(df_non_numeric.columns) != 0:
        df_dummy = dd.get_dummies(df_non_numeric, prefix_sep='.', prefix=non_numeric_columns_list)
        df = df.drop(columns=non_numeric_columns_list)
        if v_dask:
            df = dd.concat([df, df_dummy], axis=1)
        else:
            df = pd.concat([df, df_dummy], axis=1)
        print('Number of Columns for one hot encoding : ' + str(len(non_numeric_columns_list)))

    df = dask_repartition(df)

    return df