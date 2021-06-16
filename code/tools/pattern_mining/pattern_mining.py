import glob
from tools.pattern_mining.wrapper import run_itemset_exraction, mine_patterns
from tools.pattern_mining.config_pattern_mining import config_pattern_mining
from types import SimpleNamespace
import pandas as pd

def run_pattern_mining(data_path):
    c = SimpleNamespace(**config_pattern_mining) 
    files = glob.glob(data_path)
    itemsets_per_code = run_itemset_exraction(files, c)
    print('Itemsets extracted. Start pattern mining...')
    fq_itemsets, asc_rules = mine_patterns(itemsets_per_code, c)
    config_df = pd.DataFrame(config_pattern_mining, index=[0])
    excel_writer = pd.ExcelWriter('../statistics/frequent_itemsets.xlsx', engine='xlsxwriter')
    config_df.to_excel(excel_writer, sheet_name= 'config', index=False)
    fq_itemsets.to_excel(excel_writer, sheet_name= 'fq_itemsets', index=False)
    asc_rules.to_excel(excel_writer, sheet_name= 'asc_rules', index=False)
    excel_writer.save()