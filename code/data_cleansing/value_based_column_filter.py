# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from monitoring.time_it import timing
from dask.distributed import Client
import dask.dataframe as dd
import dask
# filtering based on column values: function takes Dask- or Pandas-Dataframe

def calc_unique(series):
    return series.dropna().drop_duplicates()

@timing
def value_filter(df, only_null_or_empty=True, only_one=True, only_one_and_unavailable=True, v_dask=True):
    """
    Filter out columns with constant values

    :param df: dask dataframe
    :param only_null_or_empty: remove columns with null value or no value (Binary)
    :param only_one: remove columns with one value (Binary)
    :param only_one_and_unavailable: remove columns with String non-casesensitive "Unvavailable" (Binary)
    :param v_dask: use dask dataframe format (Binary)
    :return: dask dataframe
    """

    cols2drop = []
    df = df.fillna(value=np.nan).replace('', np.nan).dropna(how='all')

    lazy_results = []
    if v_dask:
        for col in df.columns:
            series = df[col]
            lazy_result = dask.delayed(calc_unique)(series)
            lazy_results.append(lazy_result ) 
        lazy_results = dask.compute(*lazy_results)
    else:
        for col in df.columns:
            series = df[col]
            lazy_results.append(calc_unique(series))

    for col in lazy_results:
        if only_null_or_empty:
            if len(col) == 0:
                cols2drop.append(col.name)

        if only_one:
            if len(col) == 1:
                cols2drop.append(col.name)

        if only_one_and_unavailable:
            cleaned_data = []
            for v in col:
                try:
                    cleaned_data.append(v.upper())
                except:
                    cleaned_data.append(v)
            if len(cleaned_data) == 2 and 'UNAVAILABLE' in cleaned_data:
                cols2drop.append(col.name)
    if len(cols2drop) > 0:
        df = df.drop(columns=cols2drop)
    return df