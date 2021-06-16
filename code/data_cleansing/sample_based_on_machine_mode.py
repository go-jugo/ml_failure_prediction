from config import dataset
import pandas as pd
import dask.dataframe as dd

# valid machine modes
machine_modes = {
    'door_state' : ['ALL', 'UNAVAILABLE', 'CLOSED', 'UNLATCHED'],
    'door_functioning_mode' : ['ALL', 'UNAVAILABLE', 'PRODUCTION', 'SETUP'],
    'spindle_mode' : ['ALL', 'UNAVAILABLE', 'CONTOUR', 'SPINDLE'],
    'path_mode' : ['ALL','AUTOMATIC', 'MANUAL_DATA_INPUT', 'UNAVAILABLE', 'MANUAL'],
    'path_execution_state' : ['ALL', 'UNAVAILABLE', 'READY', 'STOPPED', 'ACTIVE', 'INTERRUPTED',
                               'PROGRAM_STOPPED', 'OPTIONAL_STOP', 'FEED_HOLD']   
    }

# mapping modes to columns
mapping_mode_to_cols = {
    'door_state' : ['components.door1.events.door.state'],
    'door_functioning_mode' : ['components.d1.events.functionalmode.state'],
    'spindle_mode' : ['components.c.events.rf.state', 'components.c2.events.rf2.state', 'components.c3.events.rf3.state', 'components.c4.events.rf4.state'],
    'path_mode' : ['components.path1.events.mode.state','components.path2.events.mode2.state'],
    'path_execution_state' : ['components.path1.events.exec.state','components.path2.events.exec2.state'] 
    }

selected_machine_modes = dataset['selected_machine_modes']
      
def check_machine_mode_input(user_input):
    mode_filter_dict = {}
    for mode_key, mode_values in selected_machine_modes.items():
        #User Input Validation
        if 'ALL' in mode_values and len(mode_values) > 1:
            raise ValueError('Invalid Machine Mode Input: "ALL" in combination with other inputs is not allowed')     
        for mode_value in mode_values:
            if mode_value not in machine_modes[mode_key]:
                raise ValueError('Invalid Machine Mode Input: Machine Mode "{}" does not exist for "{}"'.format(mode_value, mode_key))
        #Parse User input
        if 'ALL' in mode_values:
           pass                                 #do nothing because we do not have to filter for this mode
        else:
          mode_filter_dict[mode_key] = mapping_mode_to_cols[mode_key], mode_values
    return mode_filter_dict

def create_filter(df, mode_filter_dict):
    columns = df.columns
    df_queries = []
    global_query = ''
    if len(mode_filter_dict) == 0:
        return global_query
    # OR queries if several values are specified for certain cols
    for mode_cols, mode_values in mode_filter_dict.values():
        df_query = ''
        for col in mode_cols:
            if col not in columns:
                print('Column {} not in dataframe. No filter for this column created'.format(col))
            else:    
                for value in mode_values: 
                    if len(df_query) == 0:
                        df_query = '`{}` == "{}"'.format(col, value)
                    else:
                        df_query = df_query + ' | ' + '`{}` == "{}"'.format(col, value)
        df_queries.append(df_query)       
    # AND query to combine the OR queries 
    for query in df_queries:
        if len(global_query) == 0:
            global_query = '(' + query + ')'
        else:
            global_query = global_query + ' & (' + query + ')'          
    return global_query

def dask_filter(df_partition, query):
     df_partition_reduced = df_partition.query(query)
     return df_partition_reduced

def filter_for_machine_mode(df, selected_machine_modes = selected_machine_modes, dask = True):     
    user_input = check_machine_mode_input(selected_machine_modes)
    df_query = create_filter(df, user_input) 
    if len(df_query) == 0:
        return df
    if dask == True:
        df_reduced = df.map_partitions(dask_filter, query = df_query)
    else:
        df_reduced = df.query(df_query)
    return df_reduced