import pandas as pd
from monitoring.time_it import timing

@timing
def remove_outliers(df, number_stds=3, how = 'all'):
    print('Numer rows before outlier removal: {}'.format(len(df)))
    is_outlier_mask = (df < (df.mean() - number_stds*df.std())) | (df > (df.mean() + number_stds*df.std()))
    if how == 'all':
        rows2drop = is_outlier_mask.all(axis=1)
    else:
        rows2drop = is_outlier_mask.any(axis=1)
    df_reduced = df[~rows2drop]
    print('Numer rows after outlier removal: {}'.format(len(df_reduced)))
    return df_reduced