import pandas as pd
import dask.dataframe as dd
import numpy as np
from monitoring.time_it import timing


#function takes Dask Dataframe as input and removes Features with correlation >= correlation_coefficient
@timing
def remove_correlated_features(df, correlation_coefficient=1, dask=True):
    #reduce dataframe by not consider global timestamp and errorCodes
    df_reduced = df.select_dtypes(include='number')
    drop_errorcode_col = [col for col in df_reduced.columns if 'errorCode' in col]
    df_reduced = df_reduced.drop(columns=drop_errorcode_col)

    #create correlation matrix and determine columns to drop
    corr_matrix = df_reduced.corr().abs()
    if dask:
        corr_matrix = corr_matrix.compute()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] >= correlation_coefficient)]
    print('Number of columns to drop: ' + str(len(to_drop)))

    df = df.drop(columns=to_drop)
    return df