# -*- coding: utf-8 -*-

import pandas as pd
from monitoring.time_it import timing


# filtering based on column names

def name_filter(df, device_cols=True, name_cols=True, type_cols=True, timestamp_cols=False, sequence_number_cols=False, state_cols=False, message_cols=False):
    cols2drop = []
    regex_list = []
    if device_cols:
        regex_list.extend(['_id.$oid', 'className', 'machine', 'machineId'])
    if name_cols:
        regex_list.append('.name')
    if type_cols:
        regex_list.extend(['.componentType', '.eventType', '.type', '.subType'])
    if timestamp_cols:
        regex_list.append('timestamp.$date')
    if sequence_number_cols:
        regex_list.append('.sequence.$numberLong')
    if state_cols:
        regex_list.append('.status', '.state')
    if message_cols:
        regex_list.append('.message')

    for col in df.columns:
        for regex in regex_list:
            if regex in col:
                cols2drop.append(col)

    cols2drop = list(dict.fromkeys(cols2drop))

    df_reduced = df.drop(columns=cols2drop)
    return df_reduced