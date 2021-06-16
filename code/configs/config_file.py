"""
Modul that contains the configurations
The modul must contain a create_configs function that returns a list of dictionaries
Each dictionary in the list is a specific configuration
"""
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE, RandomOverSampler, SMOTENC
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.ensemble import EasyEnsembleClassifier

#this example creates a cartesian product of configs 
def create_configs(dataset ='CPT',target_col='components.cont.conditions.logic.errorCode'):
    base_config = dict(
        dataset=[dataset],
        sampling_frequency=['30S'],
        imputations_technique_str=['pad'],
        imputation_technique_num=['pad'],
        replace_extreme_values=[True],
        ts_fresh_window_length=[10800],
        ts_fresh_window_end=[3600],
        pw_rw_list=[[[14400,14400]]],
        ts_fresh_minimal_features=[True],
        extract_examples=[False],
        scaler=[StandardScaler()],
        target_col=[target_col],
        target_errorCode=[351],
        balance_ratio = [0.5],
        sampling_percentage = [0.5],
        random_state = [[100]],
        evaluation_metrics=['accuracy'],
        cv=[5],
        oversampling_method = [False],
        evaluate_all_data = [False],
        ml_algorithm=[RandomForestClassifier(class_weight='balanced'), SVC(kernel='linear')],
        ensemble_voting=['hard'],
        ml_list = [[SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),
                    SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),
                    SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),
                    SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True),SVC(kernel='linear', probability=True)]]
        )
    configs_pipeline = [dict(zip(base_config, v)) for v in product(*base_config.values())]
    return configs_pipeline

