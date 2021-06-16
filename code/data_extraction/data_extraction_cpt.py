from data_extraction import flatten_json_data, change_column_id
from data_merge import extract_data_for_single_machine, name_based_column_filter, merge_global_and_local_timestamps, create_global_dataframe
import re
from monitoring.time_it import timing

from os.path import splitext
buffer_data_path = '../data/Buffer_Data/'

from pathlib import Path



@timing
def data_extraction_cpt(file):
            try:
                print(file)
                df = flatten_json_data.flatton_json_data(file)
                df = change_column_id.change_column_id(df)
                df = extract_data_for_single_machine.extract_data_for_single_machine(df)
                df = name_based_column_filter.name_filter(df)
                df = merge_global_and_local_timestamps.rearrange_same_component_columns_and_create_global_artificial_timestamp_dataframe(df)
                #name = re.search(r'\d+-\d+-\d+', file).group() + str('.parquet.gzip')
                buffer_path = buffer_data_path + Path(Path(file).stem).stem +str('.parquet.gzip')
                print(buffer_path)
                df.to_parquet(buffer_path)
                #df.to_parquet( str(name))
            except Exception as e:     
                with open("..\monitoring\logs\extract.log", "a") as logf:
                    logf.write("Failed to extract {0}: {1}\n".format(str(file), str(e)))
            finally:
                pass
            return None