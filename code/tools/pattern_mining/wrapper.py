from tools.pattern_mining.itemset_preperation import extract_windows,extract_itemset_dfs, extract_error_itemsets
from tools.pattern_mining.mine_itemsets import calc_frequent_itemsets, aggregate_frequent_itemsets, aggregate_asc_rules
import pandas as pd

def extract_itemsets_per_code(df, c):
    windows = extract_windows(df, target_col=c.target_col)      
    itemset_dfs_per_code = extract_itemset_dfs(df, windows, min_gap=c.min_gap_since_last_error,
                                               min_obs=c.min_obs_since_last_error, window_len=c.window_len)     
    itemsets_per_code = extract_error_itemsets(itemset_dfs_per_code, target_col = c.target_col, cross_component=c.cross_component)      
    return itemsets_per_code

def run_itemset_exraction(files, c):
    itemsets_per_code_global = {}
    for file in files:
        data = pd.read_parquet(file) 
        error_cols = [col for col in data.columns if 'errorCode' in col]
        error_cols.append('global_timestamp')
        data = data[error_cols]
        data = data.dropna(axis=1,how='all').fillna(method='ffill')
        first_valid_loc = data.apply(lambda col: col.first_valid_index()).max()
        data = data.loc[first_valid_loc:,]     
        # itemsets per file
        itemsets_per_code = extract_itemsets_per_code(data, c)
        # aggregate itemset over all files for each error code
        for err_code, itemsets in itemsets_per_code.items():
            if err_code in itemsets_per_code_global:
                itemsets_per_code_global[err_code] = itemsets_per_code_global[err_code] + itemsets
            else:
                itemsets_per_code_global[err_code] = itemsets
    return itemsets_per_code_global

def mine_patterns(itemsets_per_code, c):
    frequent_itemsets_per_code, asc_rules_per_code = calc_frequent_itemsets(itemsets_per_code, c)
    fq_itemsets = aggregate_frequent_itemsets(frequent_itemsets_per_code)
    asc_rules = aggregate_asc_rules(asc_rules_per_code)
    return fq_itemsets, asc_rules