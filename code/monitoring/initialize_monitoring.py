import pandas as pd
import os
import shutil

def initialize_monitoring():
    file_path_monitoring = '../monitoring/' 
    df = pd.DataFrame(columns=[ 'ts', 'te', 'ts-te', 'function_name' 'memory_usage_start',  'memory_usage_end', 'data_memory'])
    df.to_csv(file_path_monitoring + 'time.csv', index=False, sep=";", decimal=",")

    dir_list = [file_path_monitoring + 'logs/', file_path_monitoring + 'debug/', file_path_monitoring + 'memory/']
    for dir in dir_list:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)