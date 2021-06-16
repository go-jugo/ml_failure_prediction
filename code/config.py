from local_conf import server
import importlib.util
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, RandomOverSampler, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier


v_dask = True                         
apply_data_extraction = False
v_dask_data_extraction = True        
debug_mode = False
write_monitoring = False              
store_results = True
buffer_data_path = '../data/Buffer_Data/*.gzip'
raw_data_path = '../data/Raw_Data/*.gz'
if server:
    raw_data_path = 'E:\Code\Fraeswerk/*.gz'


def create_configs():
    """
    set parameters of pipeline in this function to perform failure prediction with machine learning.
    This function creates a cartesian product of the parameters. Every parameter is specified in a list.
    More than one parameter can be specified in every list.

    sampling_frequency: Adjust the sampling frequency to uniform time steps, e.g. every 30 seconds (String)
    imputation_technique_str: technique to fill missing values for string columns, e.g. 'pad' performs forward fill (String)
    imputation_technique_num: technique to fill missing values for numeric columns, e.g. 'pad' performs forward fill (String)
    replace_extreme_values: replace outliers (Binary)
    ts_fresh_window_length: length of reading window in seconds (Integer)
    ts_fresh_window_end: length of prediction window in seconds (Integer)
    pw_rw_list: list of tuples for feature gerneration, 1.entry: reading window in seconds (Integer); 2.entry: prediction window in seconds (Integer)
    ts_fresh_minimal_features: only perform feature extraction for low cost features, see tsfresh documentation (Binary)
    extract_examples: generate features and buffer them in a directory (Binary)
    scaler: specify technique to standardize features (see scikit-learn documentation)
    target_col: target column for prediction of a specific errorcode (String)
    target_errorCode: errorcode for failure prediction (Integer)
    balance_ratio: ratio between failure and non-failure examples for function "extract_windows_and_features"
    sampling_percentage: ratio between failure and non-failure examples for function "eval"
    random_state: List of Integers. Every integer stands for a different data sample to perform failure prediction
    cv: k-fold crossvalidation (Integer)
    oversampling_method: specify, if oversampling method (e.g. SMOTE) should be performed. See "imbalanced-learn" documentation.
    ml_algorithm: specify ml algorithm for failure prediction. See scikit-learn documentation.
    :return: cartesian product of the parameters (dictionary)
    """
    base_config = dict(
        sampling_frequency=['30S'],
        imputations_technique_str=['pad'],
        imputation_technique_num=['pad'],
        replace_extreme_values=[True],
        ts_fresh_window_length=[3600],
        ts_fresh_window_end=[3600],
        pw_rw_list=[[[3600,3600]]],
        ts_fresh_minimal_features=[True],
        extract_examples=[False],
        scaler=[StandardScaler()],
        target_col=['components.cont.conditions.logic.errorCode'],
        target_errorCode=[351],
        balance_ratio = [0.5],
        sampling_percentage = [0.5],
        random_state = [[100]],
        cv=[5],
        oversampling_method = [False],
        evaluate_all_data = [False],
        ml_algorithm=[RandomForestClassifier(class_weight='balanced'), SVC(kernel='linear'), SVC(kernel='poly'), SVC(kernel='rbf'), SVC(kernel='sigmoid')],
        ensemble_voting=['hard'],
        ml_list = [[SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True)]]
        )
    configs_pipeline = [dict(zip(base_config, v)) for v in product(*base_config.values())]
    return configs_pipeline

configs_pipeline = create_configs()

