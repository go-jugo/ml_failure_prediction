from data_extraction import flatten_json_data, change_column_id
from data_merge import extract_data_for_single_machine, name_based_column_filter, merge_global_and_local_timestamps, create_global_dataframe
import re
from monitoring.time_it import timing

from os.path import splitext

from pathlib import Path
import pandas as pd
from config import buffer_data_path
gbt = 'global_timestamp'
def data_extraction_sitec(file):
            try:
                print(file)
                df = pd.read_excel(file) 
                df = clean_extracted_df(df, file)
                buffer_path = buffer_data_path.split('*')[0] + Path(Path(file).stem).stem +str('.parquet.gzip')
                df.to_parquet(buffer_path)
            except Exception as e:     
                with open("..\monitoring\logs\extract.log", "a") as logf:
                    logf.write("Failed to extract {0}: {1}\n".format(str(file), str(e)))
            finally:
                pass
            return None
        
def clean_extracted_df(df,file):
    df = df.rename(columns={'time':gbt,'timestamp':gbt, 'errorcode': 'errorCode'})
    df[gbt] = pd.to_datetime(df[gbt])                         
    if 'influx' in file:
        df['sensor'] = df['sensor'].astype(str) + '.samples'
        df = df.pivot(index='global_timestamp', columns='sensor', values='gesamtwirkleistung').reset_index()
    df = df.drop(columns=['main_id','Unnamed: 0'], errors='ignore').drop_duplicates() 
    cols2string = [col for col in df.columns if 'samples' not in col and gbt not in col]
    df[cols2string] = df[cols2string].astype(str)
    if 'influx' not in file:
        df['errorCode'] = df['errorCode'].astype(int)
    return df
