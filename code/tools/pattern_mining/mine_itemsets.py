import pandas as pd
from math import ceil 
import pyfpgrowth

def aggregate_frequent_itemsets(frequent_itemset_dict):
    result = pd.DataFrame(columns=['errorCode', 'numberWindowsTotal', 'itemset', 'support', 'supportRel'])
    for err_code, df in frequent_itemset_dict.items():
        result = pd.concat([result,df], axis=0)
    return result

def aggregate_asc_rules(asc_rules_dict):
    result = pd.DataFrame(columns=['errorCode', 'antecedant', 'consequent', 'confidence'])
    for err_code, df in asc_rules_dict.items():
        result = pd.concat([result,df], axis=0)
    return result

def calc_frequent_itemsets(error_itemsets_per_code, c):
    freqeuent_itemsets_per_code = {}            #contains frequent itemsets
    asc_rules_per_code = {}                     #contains association rules per code
    for err_code, error_itemsets_code_i in error_itemsets_per_code.items():
        number_windows = len(error_itemsets_code_i)        
        if number_windows == 0:
            pass 
        elif number_windows == 1:
            pass 
        else:
           patterns = pyfpgrowth.find_frequent_patterns(error_itemsets_code_i, ceil(number_windows*c.support))   
           patterns = patterns_to_pandas(err_code, patterns, number_windows)   
           if len(patterns) > 0:
               freqeuent_itemsets_per_code[err_code] = patterns
           rules = pyfpgrowth.generate_association_rules(patterns, c.confidence)
           rules = rules_to_pandas(err_code, rules)
           if len(rules) > 0:
               asc_rules_per_code[err_code] = rules 
    return freqeuent_itemsets_per_code, asc_rules_per_code

def patterns_to_pandas(error_code, pattern_dict, numberWindows):
    pandas_df = pd.DataFrame(columns=['errorCode', 'numberWindowsTotal', 'itemset', 'support', 'supportRel'])
    for itemset, support in pattern_dict.items():
        tmp_dict = {'errorCode' : error_code,
                    'numberWindowsTotal' : numberWindows,
                    'itemset' : [itemset],
                    'support' : support,
                    'supportRel' : support/numberWindows}
        tmp_df = pd.DataFrame(tmp_dict)
        pandas_df = pd.concat([pandas_df,tmp_df], axis=0)
    return pandas_df

def rules_to_pandas(error_code, rules):
    pandas_df = pd.DataFrame(columns=['errorCode', 'antecedant', 'consequent', 'confidence'])
    for antecedent, consequent in rules.items():
        print(consequent[0])
        tmp_dict = {'errorCode' : error_code,
                    'antecedant' : [antecedent],
                    'consequent' : consequent[0],
                    'confidence' : consequent[1]}
        tmp_df = pd.DataFrame(tmp_dict)
        pandas_df = pd.concat([pandas_df,tmp_df], axis=0)
    return pandas_df
