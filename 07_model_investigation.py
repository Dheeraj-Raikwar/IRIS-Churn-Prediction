'''
Author: Freddie Hampel, JMAN Group
Date: April 2021

Runs after 05_modelling.py

Script to:
1. create churn probability distributions from the test-train and out of time test dataset
2. produce comparisons between calibrated, and non-calibrated models
'''

import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score

from tools.modelling_functions import split_data_into_test_train_wrapper

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

file_loc = FileLocations()
df_feature_matrix_train = pd.read_pickle(file_loc.loc_feature_matrix_train)
df_feature_matrix_test = pd.read_pickle(file_loc.loc_feature_matrix_test)
df_feature_matrix_out_of_time_test = pd.read_pickle(file_loc.loc_feature_matrix_out_of_time_test)

col = ColumnHeaders(df_feature_matrix_train)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

# --- split into X and y --------------------------------------------------------------------------------------------- #
X_train = df_feature_matrix_train[model_config.latest_feature_list]
X_test = df_feature_matrix_test[model_config.latest_feature_list]
X_out_of_time_test = df_feature_matrix_out_of_time_test[model_config.latest_feature_list]

y_train = df_feature_matrix_train[model_config.target_col]
y_test = df_feature_matrix_test[model_config.target_col]
y_out_of_time_test = df_feature_matrix_out_of_time_test[model_config.target_col]

# --- import model and required inputs ------------------------------------------------------------------------------- #

with open(file_loc.loc_model, 'rb') as file:
    trained_model = pickle.load(file)
    file.close()

with open(file_loc.loc_model_calibrated, 'rb') as file:
    calibrated_model = pickle.load(file)
    file.close()

file_name = file_loc_tools.define_output_file_name(file_loc.loc_test_train_churn_probabilities, file_loc.dir_checks)

for name, model in [['', trained_model], ['_calibrated', calibrated_model]]:
    # predict churn probability
    results_train = model.predict_proba(X_train)
    results_test = model.predict_proba(X_test)

    X_train[col.churn_probability] = results_train[:, 1]
    X_train['test_flag'] = 0

    X_test[col.churn_probability] = results_test[:, 1]
    X_test['test_flag'] = 1

    df_results = X_train[[col.churn_probability, 'test_flag']].append(X_test[[col.churn_probability, 'test_flag']])
    df_results.reset_index(inplace=True)

    df_results[col.product_family] = df_results['index'].apply(lambda x: x.split('//')[0])
    df_results[col.product_group] = df_results['index'].apply(lambda x: x.split('//')[1])

    df_results = df_results[['index', col.churn_probability, col.product_family, col.product_group, 'test_flag']]

    df_results = pd.merge(df_results,
                          df_feature_matrix_train.reset_index()[[model_config.target_col, 'index']],
                          how='left',
                          on='index')

    # df_results.to_csv(file_loc.dir_checks + file_name + name + '.csv', index=False)
    df_results.to_csv(file_loc.loc_test_train_churn_probabilities.split('.csv')[0] + name + '.csv', index=False)

    # --- bar chart for date range in latest month file -------------------------------------------------------------- #

    out_of_time_test_predictions = model.predict_proba(X_out_of_time_test)
    df_feature_matrix_out_of_time_test['churn_prob'] = out_of_time_test_predictions[:, 1]

    roc_auc_score_out_of_time_test = roc_auc_score(df_feature_matrix_out_of_time_test[model_config.target_col],
                                                   df_feature_matrix_out_of_time_test['churn_prob'])

    print(f'roc_auc = {roc_auc_score_out_of_time_test}')

    df_feature_matrix_out_of_time_test = pd.merge(df_feature_matrix_out_of_time_test,
                                                  X_test.reset_index()[['test_flag', 'index']],
                                                  how='left',
                                                  on='index')

    df_feature_matrix_out_of_time_test['test_flag'].fillna(0, inplace=True)

    df_feature_matrix_out_of_time_test.to_csv(file_loc.dir_checks + 'march_21_predictions' + name + '.csv')

print('finished')
