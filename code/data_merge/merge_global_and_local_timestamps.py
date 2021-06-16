# -*- coding: utf-8 -*-
# necessary modules
import pandas as pd
import re
from monitoring.time_it import timing


def rearrange_same_component_columns_and_create_global_artificial_timestamp_dataframe(df):
    # if Dataframe is empty return empty Dataframe
    if (len(df) == 0):
        return pd.DataFrame()

    columns_global_all = []
    columns_components_dict_all = {}
    columns_subcategory_dict_all = {}

    # get component categories
    for col in df.columns:
        col_id_list = re.split('\.(?!\$)|\_(?!.*\.)', col)
        # global columns
        if len(col_id_list) == 1:
            columns_global_all.append(col)
        # component columns
        if len(col_id_list) == 3:
            if col_id_list[1] not in columns_components_dict_all:
                columns_components_dict_all[col_id_list[1]] = list()
                columns_components_dict_all[col_id_list[1]].append(col)
            else:
                columns_components_dict_all[col_id_list[1]].append(col)
        # subcategory columns (nested list)
        if len(col_id_list) > 3:
            if len(col_id_list) == 5:
                if (col_id_list[1], col_id_list[2], col_id_list[3]) not in columns_subcategory_dict_all:
                    columns_subcategory_dict_all[(col_id_list[1], col_id_list[2], col_id_list[3])] = [[]]
                    columns_subcategory_dict_all[(col_id_list[1], col_id_list[2], col_id_list[3])][0].append(col)
                else:
                    columns_subcategory_dict_all[(col_id_list[1], col_id_list[2], col_id_list[3])][0].append(col)
            if len(col_id_list) > 5:
                if len(columns_subcategory_dict_all[(col_id_list[1], col_id_list[2], col_id_list[3])]) < int(
                        col_id_list[5]) + 1:
                    columns_subcategory_dict_all[(col_id_list[1], col_id_list[2], col_id_list[3])].append(list())
                    columns_subcategory_dict_all[(col_id_list[1], col_id_list[2], col_id_list[3])][
                        int(col_id_list[5])].append(col)
                else:
                    columns_subcategory_dict_all[(col_id_list[1], col_id_list[2], col_id_list[3])][
                        int(col_id_list[5])].append(col)

    all_dataframes_subcategories = {}
    global_timestamp_list = []

    # extract dataframes of subcategories, sort them and save them in dict
    for key in columns_subcategory_dict_all:
        df_subcategory = pd.DataFrame()
        for element in columns_subcategory_dict_all[key]:
            df_subcategory_add = df[element]
            df_subcategory_add = df_subcategory_add.dropna(axis=0, how='all')
            df_subcategory_add = df_subcategory_add.rename(columns=lambda x: re.sub(r'\_\d+$', '', x))
            df_subcategory = pd.concat([df_subcategory, df_subcategory_add], axis=0)
        sort_criteria_1 = [col for col in df_subcategory.columns if 'timestamp.$date' in col][0]
        sort_criteria_2 = [col for col in df_subcategory.columns if 'sequence.$numberLong' in col][0]
        df_subcategory[sort_criteria_2] = df_subcategory[sort_criteria_2].map(float)
        df_subcategory_sorted = df_subcategory.sort_values(by=[sort_criteria_1, sort_criteria_2])
        df_subcategory_sorted = df_subcategory_sorted.drop_duplicates(subset=sort_criteria_1, keep='last').reset_index(
            drop=True)
        global_timestamp_list.extend(df_subcategory_sorted[sort_criteria_1].to_list())
        global_timestamp_list = list(set(global_timestamp_list))
        all_dataframes_subcategories[key] = df_subcategory_sorted

    print('Number global timestamps before merging subcategories: {}'.format(len(global_timestamp_list)))

    # create global dataframe and merge on local timestamp of subcategories
    df_global = pd.DataFrame(global_timestamp_list, columns={'global_timestamp'}).sort_values(
        by=['global_timestamp']).reset_index(drop=True)
    for key in all_dataframes_subcategories:
        timestamp = [col for col in all_dataframes_subcategories[key].columns if 'timestamp.$date' in col][0]
        df_global = df_global.merge(all_dataframes_subcategories[key], left_on='global_timestamp',
                                    right_on=str(timestamp), how='left')

    print('Number global timestamps after merging subcategories: {}'.format(len(df_global)))

    # drop local timestamp columns
    local_timestamp_columns = [col for col in df_global.columns if 'timestamp.$date' in col]
    if len(local_timestamp_columns) == len(columns_subcategory_dict_all):
        df_global = df_global.drop(columns=local_timestamp_columns)
    else:
        raise ValueError('Number of local component timestamps and number of component subcategories are not the same')

    # drop local sequenceNumber columns
    local_sequenceNumber_columns = [col for col in df_global.columns if 'sequence.$numberLong' in col]
    if len(local_sequenceNumber_columns) == len(columns_subcategory_dict_all):
        df_global = df_global.drop(columns=local_sequenceNumber_columns)
    else:
        raise ValueError(
            'Number of local component SequenceNumbers and number of component subcategories are not the same')
    # get first valid index
    first_valid_index = df_global.apply(lambda col: col.first_valid_index()).min()
    df_global = df_global.loc[first_valid_index:].reset_index(drop=True)
    return df_global


