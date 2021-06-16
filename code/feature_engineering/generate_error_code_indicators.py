import pandas as pd
import dask.dataframe as dd
import dask
import numpy as np
from monitoring.time_it import timing
from tools.series_list_to_df import series_list_to_df


def cumsum_until_zero(series):
    return series.groupby(series.eq(0).cumsum()).cumsum()


@timing
def gen_error_code_indicators(df, relevant_col, relevant_err_code, v_dask=True):

    if v_dask == True:

        df_error_col = df.categorize(columns=[relevant_col])[relevant_col]
        df_dummy = dd.get_dummies(df_error_col, prefix_sep='.', prefix=relevant_col)
        column_name_relevant_err_code = str(relevant_col) + '.' + str(relevant_err_code) + '.0'
        column_name_default_err_code = str(relevant_col) + '.' + str(-2147483648) + '.0'
        column_name_list = [column_name_relevant_err_code, column_name_default_err_code]

        lazy_results = []
        for element in column_name_list:
            series = df_dummy[element]
            name = str(series.name[:-2]) + '.consecutiveErrors'
            series = series.rename(name)
            lazy_result = dask.delayed(cumsum_until_zero)(series)
            lazy_results.append(lazy_result)
        series_collection = dask.compute(*lazy_results)
        df_consecutive = series_list_to_df(series_collection, v_dask)

        df = dd.concat([df, df_consecutive], axis=1)

        return df






'''
@timing
def gen_error_code_indicators(df, relevant_col, relevant_err_code, v_dask = True):
     
    error_cols = [col for col in df.columns if '.errorCode' in col]
    
    if v_dask == True:
        for col in error_cols:
            error_codes = df[col].compute().replace('', np.nan).dropna().unique()     
            #check input
            if relevant_err_code not in error_codes and relevant_col == col:
                raise ValueError('Error Code {} not in {}'.format(relevant_err_code, relevant_col))        
            for error_code in error_codes:        
                if col == relevant_col and error_code == relevant_err_code:         
                    col_name = str(col) + '_' + str(int(error_code)) + '_' + 'number_in_sucession'
                    indicator_series = (df[col].compute() == error_code).astype(int)
                    helper = indicator_series.cumsum()
                    df[col_name] = helper.sub(helper.mask(indicator_series != 0).ffill(), fill_value=0).astype(int)
                else:
                    col_name = str(col) + '_' + str(int(error_code)) + '_' + 'occured'
                    indicator_series = (df[col].compute() == error_code).astype(int)
                    df[col_name] = indicator_series                  
            #drop original error code col
            df = df.drop(columns=[col])
        return df
     
    if v_dask == False:
       for col in error_cols:
            error_codes = df[col].replace('', np.nan).dropna().unique()        
            #check input
            if relevant_err_code not in error_codes and relevant_col == col:
                raise ValueError('Error Code {} not in {}'.format(relevant_err_code, relevant_col))   
            for error_code in error_codes:              
                if col == relevant_col and error_code == relevant_err_code:                         
                    col_name = str(col) + '_' + str(int(error_code)) + '_' + 'number_in_sucession'
                    indicator_series = (df[col] == error_code).astype(int)
                    helper = indicator_series.cumsum()
                    df[col_name] = helper.sub(helper.mask(indicator_series != 0).ffill(), fill_value=0).astype(int)
                else:
                    col_name = str(col) + '_' + str(int(error_code)) + '_' + 'occured'
                    indicator_series = (df[col] == error_code).astype(int)
                    df[col_name] = indicator_series          
            #drop original error code col
            df.drop(columns=[col], inplace=True)
       return df

'''


