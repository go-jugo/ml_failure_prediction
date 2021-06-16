import itertools
import pandas as pd
pd.options.mode.chained_assignment = None

def pairwise(iterable):
    #s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def extract_windows(df, target_col = 'components.cont.conditions.logic.errorCode'):
    errors_unique = df[target_col].unique()
    tmp_df = df.loc[:,['global_timestamp', target_col]].copy()
    # for each error extract the event windows -i.e. the times the error occurs
    event_windows = {}
    for error in errors_unique:
        col = 'error' + '_' + str(int(error)) + '_' + 'number_in_sucession'           
        indicator_series = (df[target_col] == error).astype(int)     
        helper = indicator_series.cumsum()
        tmp_df[col] = helper.sub(helper.mask(indicator_series != 0).ffill(), fill_value=0).astype(int)     
        timestamps_error_i = tmp_df[tmp_df[col] !=0]
        timestamps_error_i['dif'] = (timestamps_error_i[col].shift(-1) - timestamps_error_i[col])
        # start of each window
        starts = timestamps_error_i[timestamps_error_i[col] == 1]['global_timestamp'].tolist()
        # end of each window
        ends = timestamps_error_i[(timestamps_error_i['dif'] <= 0)|(timestamps_error_i['dif'].isna())]['global_timestamp'].tolist()
        event_windows[error] = [[start,end] for start,end in zip(starts, ends)]
        tmp_df.drop(columns=[col], inplace = True)
    del event_windows[-2147483648]                      #OK error code not relevant
    return event_windows

# there should be at least 10 obs and 300sec to the same last error Code, window lengths extracted for itemset extraction: 300sec
def extract_itemset_dfs(df, error_dict, min_gap = 300, min_obs = 10, window_len=300): 
    cols = [col for col in df.columns if '.errorCode' in col]
    cols.append('global_timestamp')
    db = {}
    for error, error_windows in error_dict.items():
        dfs_before_error_i = []
        if len(error_windows) == 1:         #if only one event window append the df till this event
            start_window = error_windows[0][0]
            #fetch the window where the event does no occur --> 5 min before
            window = df[(start_window > df['global_timestamp']) & (df['global_timestamp'] >= start_window - pd.Timedelta(seconds=window_len))]
            if len(window) > min_obs:
                    dfs_before_error_i.append(window[cols])
        if len(error_windows) > 1:
            #if more than one window append the first event window before the first error
            start_window = error_windows[0][0]
            window = df[(start_window > df['global_timestamp']) & (df['global_timestamp'] >= start_window - pd.Timedelta(seconds=window_len))]
            if len(window) > min_obs:
                    dfs_before_error_i.append(window[cols])
            #if more than one window check that the next error occurs at least 5min after the previous error
            for pair in pairwise(error_windows):
                end_antecedent_win = pair[0][1]
                start_subsequent_win = pair[1][0]
                if (start_subsequent_win - end_antecedent_win).seconds > min_gap:
                    window = df[(start_subsequent_win > df['global_timestamp']) & (df['global_timestamp'] >= start_subsequent_win - pd.Timedelta(seconds=window_len))]
                    if len(window) > min_obs:
                        dfs_before_error_i.append(window[cols])
        db[error]=dfs_before_error_i
    return db
            
def extract_error_itemsets(dfs_dict, target_col = 'components.cont.conditions.logic.errorCode', cross_component=False):
    error_itemsets_per_code = {}
    if cross_component == False: 
        for error, error_dfs in dfs_dict.items():
            error_itemsets_code_i = []
            for error_df in error_dfs:
                error_itemset = error_df[target_col].tolist()
                error_itemset = list(set(error_itemset))                              # remove duplicates
                error_itemset = [int(x) for x in error_itemset if x != -2147483648]   # remove all OK error Codes
                error_itemsets_code_i.append(sorted(error_itemset))
            error_itemsets_per_code[error] = error_itemsets_code_i
    if cross_component == True:
        for error, error_dfs in dfs_dict.items():
            error_itemsets_code_i = []
            for error_df in error_dfs:
                 error_cols = [col for col in error_df.columns if '.errorCode' in col]
                 error_itemset = error_df[error_cols].values.flatten()
                 error_itemset = list(set(error_itemset))                              # remove duplicates
                 error_itemset = [int(x) for x in error_itemset if x != -2147483648]   # remove all OK error Codes
                 error_itemsets_code_i.append(sorted(error_itemset))
            error_itemsets_per_code[error] = error_itemsets_code_i
    return error_itemsets_per_code