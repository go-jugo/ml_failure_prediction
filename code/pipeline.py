import pandas as pd
import re
from os.path import splitext
from types import SimpleNamespace
import dask.dataframe as dd
from data_merge import create_global_dataframe
from data_cleansing import impute_missing_values, value_based_column_filter, one_hot_encode_categories, remove_correlated_features, remove_error_codes
from feature_engineering import standardize_features, generate_error_code_indicators
from feature_engineering.extract_windows_and_engineer_features_with_tsfresh import extract_windows_and_features
from feature_engineering.extract_examples import extract_examples
from data_cleansing.adjust_sampling_frequency import adjust_sampling_frequency
from ml_evaluation.get_x_y import get_x_y
from ml_evaluation.eval import eval
from ml_evaluation.eval_ensemble import eval_ensemble
from feature_engineering.remove_global_timestamp import remove_global_timestamp
from multiprocessing import Pool
from config import v_dask_data_extraction, v_dask, apply_data_extraction, buffer_data_path, raw_data_path
from local_conf import pool_size, draw_sample, sample_size
from feature_engineering.create_error_code_col import create_error_code_col
from data_extraction.select_files import select_files
from data_extraction.data_extraction_schedule import data_extraction_schedule


def run_pipeline(config):
    c = SimpleNamespace(**config)
    if apply_data_extraction:
      raw_files = select_files(raw_data_path, buffer_data_path)
      data_extraction_schedule(v_dask_data_extraction, raw_files, pool_size, c.dataset)
    print('2. Data merge')
    df = create_global_dataframe.create_global_dataframe(buffer_data_path, draw_sample, sample_size)
    print('3. Data cleansing')
    df = value_based_column_filter.value_filter(df, v_dask=v_dask)
    error_code_series = create_error_code_col(df, c.target_col)
    df = adjust_sampling_frequency(df, sampling_frequency=c.sampling_frequency, v_dask=v_dask)
    df = impute_missing_values.impute_missing_values(df, v_dask=v_dask, replace_extreme_values=c.replace_extreme_values,
                                                     string_imputation_method=c.imputations_technique_str,
                                                     numeric_imputation_method=c.imputation_technique_num)
    df = impute_missing_values.slice_valid_data(df, v_dask=v_dask)
    df = one_hot_encode_categories.one_hot_encode_categories(df, errorcode_col=c.target_col, v_dask=v_dask)
    df = remove_error_codes.remove_error_codes(df, dependent_variable=c.target_col)
    print('4. Feature engineering')
    df = extract_examples(df, error_code_series, errorcode_col=c.target_col, errorcode=c.target_errorCode,
                          pw_rw_list=c.pw_rw_list, minimal_features=c.ts_fresh_minimal_features,
                          extract_examples=c.extract_examples)
    df = extract_windows_and_features(df, error_code_series, errorcode_col=c.target_col,
                                      errorcode=c.target_errorCode, window_length=c.ts_fresh_window_length,
                                      window_end=c.ts_fresh_window_end, balance_ratio=c.balance_ratio,
                                      minimal_features=c.ts_fresh_minimal_features, v_dask=v_dask)
    print('5. ML evaluation')
    df = standardize_features.standardize_features(df, errorcode_col=c.target_col, scaler=c.scaler)
    df = remove_global_timestamp(df)
    df, y = get_x_y(df, target_variables_column=c.target_col, errorcode=c.target_errorCode)
    scores = eval(df, y, config=config, crossvalidation=c.cv, clf=c.ml_algorithm,
                  sampling_percentage=c.sampling_percentage, random_state=c.random_state)
    print(scores)