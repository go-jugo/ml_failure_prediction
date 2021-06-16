import glob
import pandas as pd
import dask.dataframe as dd
from monitoring.time_it import timing
from tools.dask_repartition import dask_repartition
import random

@timing
def create_global_dataframe(buffer_data_path, draw_sample=False, sample_size=0):
    """
    Create a global dask dataframe from parquet files

    :param buffer_data_path: path to files (String)
    :param draw_sample: specify if only a subset of files is used. Recommended for testing (Binary)
    :param sample_size: Number of files for creating global dataframe. Relevant, if draw_sample = True. (Integer)
    :return: Global dask dataframe
    """

    files = glob.glob(buffer_data_path)
    if draw_sample:
        files = random.sample(files, sample_size)
    cols2read = compute_col_union(files)
    df = dd.read_parquet(files, columns=cols2read)
    df = dask_repartition(df)
    print('Setting index ...')
    df = df.set_index('global_timestamp')
    df = dask_repartition(df)
    df['global_timestamp'] = df.index
    print('Removing duplicates ...')
    df = df.map_partitions(lambda d: d.drop_duplicates(subset='global_timestamp'))
    df = df.drop('global_timestamp', axis=1)
    df = dask_repartition(df)
    object_column_list = list(df.select_dtypes(include='object').columns)
    df[object_column_list] = df[object_column_list].astype(str)
    return df


def compute_col_union(file_path_list):
    cols_global = []
    for file in file_path_list:
        cols_global.append(list(dd.read_parquet(file).columns))
    cols_union = sorted(list(set(cols_global[0]).intersection(*cols_global)))
    return cols_union