import pandas as pd
import numpy as np
from monitoring.time_it import timing


def change_column_id(df):
    all_non_component_columns = [col for col in df.columns if not col.startswith('components.')]
    dict_component = {}
    dict_component['global'] = df[all_non_component_columns]
    component_id_columns = [col for col in df.columns if (col.startswith('components.') & col.endswith('.id') & (col.count('.') == 2))]
    id_count = 0

    for element in component_id_columns:
        id_count += 1
        component_name = element[:-2]
        component_columns = [col for col in df.columns if component_name in col]
        dict_component[component_name] = df[component_columns].copy()
        dict_component[component_name] = dict_component[component_name].pivot(columns=element)
        if dict_component[component_name].columns.get_level_values(1).isnull().any():
            for entry in dict_component[component_name].columns:
                if np.nan in entry:
                    dict_component[component_name] = dict_component[component_name].drop(entry, axis=1)
        dict_component[component_name].columns = ['_components.'.join(col) + '.' for col in dict_component[component_name].columns.values]
        subcategory_ids = [id for id in dict_component[component_name].columns if 'id' in id]
        df_subcategory_overall = pd.DataFrame()
        for col in subcategory_ids:
            id_count += 1
            subcategory_name = col[:col.find('id')]
            component_id = col[col.find('_components.'):]
            subcategory_name_cleaned = ''.join([i for i in subcategory_name.replace('components', '').replace('.', '') if not i.isdigit()])
            subcategory_columns = [x for x in dict_component[component_name].columns if ((subcategory_name in x) and (component_id in x))]
            df_subcategory = dict_component[component_name][subcategory_columns].copy()
            df_subcategory = df_subcategory.pivot(columns=col)
            if df_subcategory.columns.get_level_values(1).isnull().any():
                for entry in df_subcategory.columns:
                    if np.nan in entry:
                        df_subcategory = df_subcategory.drop(entry, axis=1)
            df_subcategory.columns = [str(subcategory_name_cleaned + '.').join(x) for x in df_subcategory.columns.values]
            df_subcategory.columns = df_subcategory.columns.str.replace(subcategory_name, '')
            df_subcategory.columns = [x[x.find('_') + 1:] + '.' + x[:x.find('_')] for x in df_subcategory.columns]
            df_subcategory_overall = pd.concat([df_subcategory_overall, df_subcategory], axis=1)
            dict_component[component_name] = dict_component[component_name].drop(columns=subcategory_columns)
        dict_component[component_name].columns = dict_component[component_name].columns.str.replace(component_name, '')
        dict_component[component_name].columns = [x[x.find('_') + 1:] + x[:x.find('_')] for x in dict_component[component_name].columns]
        dict_component[component_name] = pd.concat([dict_component[component_name], df_subcategory_overall], axis=1)
    overall_df = pd.concat(dict_component, axis=1)
    overall_df.columns = overall_df.columns.droplevel()
    overall_df_cols = pd.Series(overall_df.columns)
    for dup in overall_df_cols[overall_df_cols.duplicated()].unique():
        overall_df_cols[overall_df_cols[overall_df_cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(overall_df_cols == dup))]
    overall_df.columns = overall_df_cols
    if len(df.columns) > (len(overall_df.columns) + id_count):
        raise ValueError('Number of columns in new Dataframe is not consistent. Difference = ' + (str(len(df.columns) - (len(overall_df.columns) + id_count))))
    return overall_df