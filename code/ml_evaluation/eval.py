from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn import datasets
import sklearn 
import statistics
import pandas as pd
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn import svm
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier
from monitoring.time_it import timing
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import xlsxwriter
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.svm import LinearSVC, SVC
from openpyxl import load_workbook
from types import SimpleNamespace
import ast
import glob
from math import ceil

excel_weights_path = '../results/weights/svm'


@timing
def eval(X, y, config, crossvalidation, clf, sampling_percentage, random_state):
    """
    Evaluate Machine Learning Algorithms on the dataset and save results in xlsx-file

    :param X: feature matrix (pandas dataframe)
    :param y: dependet variable (pandas dataframe)
    :param config: configuration file (dictionary)
    :param crossvalidation: k-fold crossvalidation (Integer)
    :param clf: machine learning algorithm (scikit-learn)
    :param sampling_percentage: ratio between failures and non-failures (float)
    :param random_state: list of integers, where every Integer stands for a different data sample to perform the
    machine learning evaluation (List)
    :return: scores (dictionary)
    """

    print('Configurations: ' + str(config))

    def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[config['target_errorCode'], -1])[0, 0]
    def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[config['target_errorCode'], -1])[1, 1]
    def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[config['target_errorCode'], -1])[1, 0]
    def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=[config['target_errorCode'], -1])[0, 1]

    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score, pos_label=config['target_errorCode']),
               'recall': make_scorer(recall_score, pos_label=config['target_errorCode']),
               'f1_score': make_scorer(f1_score, pos_label=config['target_errorCode']),
               'precision_neg': make_scorer(precision_score, pos_label=-1),
               'recall_neg': make_scorer(recall_score, pos_label=-1),
               'f1_score_neg': make_scorer(f1_score, pos_label=-1),
               'tp': make_scorer(tp),
               'tn': make_scorer(tn),
               'fp': make_scorer(fp),
               'fn': make_scorer(fn)}

    number_of_errors = len(list(y[y == config['target_errorCode']].index))
    number_of_non_errors = int(ceil(((number_of_errors / sampling_percentage) - number_of_errors)))

    for state in random_state:
        rus = RandomUnderSampler(random_state=state, sampling_strategy={config['target_errorCode']: number_of_errors,
                                                                        -1: number_of_non_errors})
        X_rus, y_rus = rus.fit_resample(X, y)
        X_remain = X.drop(index=rus.sample_indices_)
        y_remain = y.drop(index=rus.sample_indices_)

        cv = StratifiedKFold(n_splits=crossvalidation, shuffle=True, random_state=42)

        scores = {}
        for element in scoring:
            scores['test_' + element] = []

        if config['oversampling_method']:
            for train_idx, test_idx, in cv.split(X_rus, y_rus):
                X_train, y_train = X_rus.iloc[train_idx], y_rus.iloc[train_idx]
                X_test, y_test = X_rus.iloc[test_idx], y_rus.iloc[test_idx]
                oversample = config['oversampling_method']
                X_train_oversampled, y_train_oversampled = oversample.fit_sample(X_train, y_train)
                clf.fit(X_train_oversampled, y_train_oversampled)
                y_pred = clf.predict(X_test)
                scores_dict = classification_report(y_test, y_pred, output_dict=True)
                scores['test_accuracy'].append(scores_dict['accuracy'])
                scores['test_precision'].append(scores_dict[str(config['target_errorCode'])]['precision'])
                scores['test_precision_neg'].append(scores_dict['-1']['precision'])
                scores['test_recall'].append(scores_dict[str(config['target_errorCode'])]['recall'])
                scores['test_recall_neg'].append(scores_dict['-1']['recall'])
                scores['test_f1_score'].append(scores_dict[str(config['target_errorCode'])]['f1-score'])
                scores['test_f1_score_neg'].append(scores_dict['-1']['f1-score'])
                scores['test_tp'].append(tp(y_test, y_pred))
                scores['test_tn'].append(tn(y_test, y_pred))
                scores['test_fp'].append(fp(y_test, y_pred))
                scores['test_fn'].append(fn(y_test, y_pred))
            for element in scores:
                scores[element] = np.array(scores[element])
        elif config['evaluate_all_data']:
            for train_index, test_index in cv.split(X_rus, y_rus):
                X_train = X_rus.iloc[train_index]
                y_train = y_rus.iloc[train_index]
                clf.fit(X_train, y_train)
                X_test = X_rus.iloc[test_index]
                X_test = pd.concat([X_test, X_remain], axis=0)
                y_test = y_rus.iloc[test_index]
                y_test = pd.concat([y_test, y_remain], axis=0)
                y_pred = clf.predict(X_test)
                scores_dict = classification_report(y_test, y_pred, output_dict=True)
                scores['test_accuracy'].append(scores_dict['accuracy'])
                scores['test_precision'].append(scores_dict[str(config['target_errorCode'])]['precision'])
                scores['test_precision_neg'].append(scores_dict['-1']['precision'])
                scores['test_recall'].append(scores_dict[str(config['target_errorCode'])]['recall'])
                scores['test_recall_neg'].append(scores_dict['-1']['recall'])
                scores['test_f1_score'].append(scores_dict[str(config['target_errorCode'])]['f1-score'])
                scores['test_f1_score_neg'].append(scores_dict['-1']['f1-score'])
                scores['test_tp'].append(tp(y_test, y_pred))
                scores['test_tn'].append(tn(y_test, y_pred))
                scores['test_fp'].append(fp(y_test, y_pred))
                scores['test_fn'].append(fn(y_test, y_pred))
            for element in scores:
                scores[element] = np.array(scores[element])
        else:
            scores = cross_validate(clf.fit(X_rus, y_rus), X=X_rus, y=y_rus, cv=cv, scoring=scoring, return_estimator=False)
            print('Evaluation with crossvalidation')

        for key, value in scores.items():
            print(str(key))
            print(value)
            print('M: ' + str(value.mean()))
            print('SD: ' + str(value.std()))

        wb = load_workbook(filename='../results/Evaluation_results_rus.xlsx')
        ws1 = wb.active
        counter = len(list(ws1.rows))
        n = len(X)
        n_error_class = len(y_rus[y_rus == config['target_errorCode']])
        n_non_error_class = len(y_rus[y_rus != config['target_errorCode']])
        sampling_frequency = config['sampling_frequency']
        imputations_technique_str = config['imputations_technique_str']
        imputation_technique_num = config['imputation_technique_num']
        ts_fresh_window_length = config['ts_fresh_window_length']
        ts_fresh_window_end = config['ts_fresh_window_end']
        ts_fresh_minimal_features = config['ts_fresh_minimal_features']
        target_col = config['target_col']
        target_errorCode = config['target_errorCode']
        rand_state = state
        sampling_percentage = config['sampling_percentage']
        balance = config['balance_ratio']
        oversampling = str(config['oversampling_method'])
        ml_algorithm = str(config['ml_algorithm'])
        cv = config['cv']
        Accuracy = str(scores['test_accuracy'])
        Accuracy_mean = scores['test_accuracy'].mean()
        Accuracy_std = scores['test_accuracy'].std()
        Precision = str(scores['test_precision'])
        Precision_neg = str(scores['test_precision_neg'])
        Precision_mean = scores['test_precision'].mean()
        Precision_mean_neg = scores['test_precision_neg'].mean()
        Precision_std = scores['test_precision'].std()
        Precision_std_neg = scores['test_precision_neg'].std()
        Recall = str(scores['test_recall'])
        Recall_neg = str(scores['test_recall_neg'])
        Recall_mean = scores['test_recall'].mean()
        Recall_mean_neg = scores['test_recall_neg'].mean()
        Recall_std = scores['test_recall'].std()
        Recall_std_neg = scores['test_recall_neg'].std()
        F1_Score = str(scores['test_f1_score'])
        F1_Score_neg = str(scores['test_f1_score_neg'])
        F1_Score_mean = scores['test_f1_score'].mean()
        F1_Score_mean_neg = scores['test_f1_score_neg'].mean()
        F1_Score_std = scores['test_f1_score'].std()
        F1_Score_std_neg = scores['test_f1_score_neg'].std()
        tp_cv = str(scores['test_tp'])
        tn_cv = str(scores['test_tn'])
        fp_cv = str(scores['test_fp'])
        fn_cv = str(scores['test_fn'])
        tp_sum = scores['test_tp'].sum()
        tn_sum = scores['test_tn'].sum()
        fp_sum = scores['test_fp'].sum()
        fn_sum = scores['test_fn'].sum()
        tp_mean = scores['test_tp'].mean()
        tn_mean = scores['test_tn'].mean()
        fp_mean = scores['test_fp'].mean()
        fn_mean = scores['test_fn'].mean()

        list_to_write_to_file = [counter, n, n_error_class, n_non_error_class, sampling_frequency,
                                 imputations_technique_str, imputation_technique_num, ts_fresh_window_length,
                                 ts_fresh_window_end, ts_fresh_minimal_features, target_col, target_errorCode,
                                 rand_state,
                                 sampling_percentage, balance, oversampling,
                                 ml_algorithm, cv, Accuracy, Accuracy_mean, Accuracy_std, Precision, Precision_neg,
                                 Precision_mean, Precision_mean_neg,
                                 Precision_std, Precision_std_neg, Recall, Recall_neg, Recall_mean, Recall_mean_neg,
                                 Recall_std,
                                 Recall_std_neg, F1_Score, F1_Score_neg, F1_Score_mean, F1_Score_mean_neg, F1_Score_std,
                                 F1_Score_std_neg,
                                 tp_cv, tn_cv, fp_cv, fn_cv, tp_sum, tn_sum, fp_sum, fn_sum, tp_mean, tn_mean, fp_mean,
                                 fn_mean]

        ws1.append(list_to_write_to_file)

        wb.save(filename='../results/Evaluation_results_rus.xlsx')

    return scores





