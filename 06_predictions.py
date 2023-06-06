'''
Author: Freddie Hampel, JMAN Group
Date: April 2021

Runs after 05_modelling.py

Script to:
1. Predict for the datasets. TODO change to run for the latest monthe when needed for operation
2. Produce shapley values for feature contributions to that prediction
3. Produce the same results for the full dataset if required for investigation
'''

from catboost import CatBoostClassifier
import pickle
from math import exp

import pandas as pd
import numpy as np

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools, write_dfs_to_excel
from tools.modelling_functions import predict_and_extract_shapley_values

file_loc = FileLocations()
df_feature_matrix_out_of_time_test = pd.read_pickle(file_loc.loc_feature_matrix_out_of_time_test)

col = ColumnHeaders(df_feature_matrix_out_of_time_test)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

ctrl_run_shap_for_test_train_set = True

df_feature_matrix_out_of_time_test = df_feature_matrix_out_of_time_test[
    model_config.latest_feature_list + [col.parent_id, model_config.target_col]]


# --- funtion for repetition ----------------------------------------------------------------------------------------- #

def melt_and_merge_shap_and_features(df_shap_values, list_shap_columns, join_col='index'):
    df_output_shap_melt = df_shap_values.melt(
        id_vars=[join_col] + ['churn_probability', 'expected_value'],
        value_vars=list_shap_columns,
        var_name='feature',
        value_name='shap_value',
        ignore_index=True)

    df_output_features_melt = df_shap_values.melt(
        id_vars=[join_col] + [id_vars for id_vars in col.list_id_vars if id_vars in df_shap_values.columns],
        value_vars=list_features,
        var_name='feature',
        value_name='feature_value',
        ignore_index=True)

    df_output_shap_melt['feature'] = df_output_shap_melt['feature'].str.replace('shap_', '')

    df_output_shap_melt = pd.merge(df_output_shap_melt,
                                   df_output_features_melt,
                                   how='left',
                                   on=[join_col, 'feature'])

    return df_output_shap_melt


# --- import model and required inputs ------------------------------------------------------------------------------- #

with open(file_loc.loc_model, 'rb') as file:
    trained_model = pickle.load(file)
    file.close()

with open(file_loc.loc_model_calibrated, 'rb') as file:
    calibrated_model = pickle.load(file)
    file.close()

list_features = [*df_feature_matrix_out_of_time_test.drop(columns=[col.parent_id, model_config.target_col]).columns]

X_out_of_time = df_feature_matrix_out_of_time_test[list_features]

df_shap_values, list_shap_columns = predict_and_extract_shapley_values(trained_model, calibrated_model,
                                                                       X_out_of_time)

df_feature_matrix_out_of_time_test.reset_index(inplace=True)

df_shap_values = df_feature_matrix_out_of_time_test.reset_index().join(df_shap_values)

df_output_shap_melt = melt_and_merge_shap_and_features(df_shap_values,
                                                       list_shap_columns)
df_output_shap_melt['abs_shap_value'] = df_output_shap_melt['shap_value'].abs()

file_name = file_loc_tools.define_output_file_name(file_loc.loc_predictions_with_shap_values,
                                                   file_loc.dir_output)

df_shap_values.to_csv(file_loc.dir_output + file_name + '.csv', index=False)
df_shap_values.to_csv(file_loc.loc_predictions_with_shap_values, index=False)

file_name_2 = file_loc_tools.define_output_file_name(file_loc.loc_predictions_with_shap_values_melt,
                                                     file_loc.dir_output)

df_output_shap_melt.to_csv(file_loc.dir_output + file_name_2 + '.csv', index=False)
df_output_shap_melt.to_csv(file_loc.loc_predictions_with_shap_values_melt,
                           index=False)  # save without version number for tableau
df_av_shap = df_output_shap_melt.groupby('feature')[['shap_value', 'abs_shap_value']].mean()

if ctrl_run_shap_for_test_train_set:

    df_feature_matrix_train = pd.read_pickle(file_loc.loc_feature_matrix_train)
    df_feature_matrix_test = pd.read_pickle(file_loc.loc_feature_matrix_test)

    df_feature_matrix_train['test_flag'] = 0
    df_feature_matrix_test['test_flag'] = 1

    df_feature_matrix_test_train = df_feature_matrix_train.append(df_feature_matrix_test)

    df_test_train_flags = df_feature_matrix_test_train.reset_index()[['index', 'test_flag']]

    df_feature_matrix_test_train = df_feature_matrix_test_train[
        model_config.latest_feature_list + [col.parent_id, model_config.target_col]]

    X_full = df_feature_matrix_test_train[list_features]

    # extract id features from index
    df_feature_matrix_test_train.reset_index(inplace=True)

    for n, x in enumerate(col.list_id_vars):
        df_feature_matrix_test_train[x] = df_feature_matrix_test_train['index'].apply(lambda x: x.split('//')[n])

    df_feature_matrix_test_train.set_index('index', inplace=True)

    df_shap_values_full, _ = predict_and_extract_shapley_values(trained_model, calibrated_model, X_full)

    df_feature_matrix_test_train.reset_index(inplace=True)

    df_shap_values_full = df_feature_matrix_test_train.join(df_shap_values_full)

    df_output_shap_melt_full = melt_and_merge_shap_and_features(df_shap_values_full,
                                                                list_shap_columns)
    df_output_shap_melt_full['abs_shap_value'] = df_output_shap_melt_full['shap_value'].abs()

    df_output_shap_melt_full = pd.merge(df_output_shap_melt_full,
                                        df_test_train_flags,
                                        how='left',
                                        on='index')

    file_name_3 = file_loc_tools.define_output_file_name(file_loc.loc_full_dataset_predictions_with_shap_values,
                                                         file_loc.dir_output)

    df_output_shap_melt_full.to_csv(file_loc.dir_output + file_name_3 + '.csv', index=False)
    df_output_shap_melt_full.to_csv(file_loc.loc_full_dataset_predictions_with_shap_values, index=False)

    df_av_shap_full = df_output_shap_melt_full.groupby('feature')[['shap_value', 'abs_shap_value']].mean()

    df_av_shap_full.rename({'shap_value': 'test_train_average_shap'}, axis=1, inplace=True)
    df_av_shap.rename({'shap_value': 'mar_21_average_shap'}, axis=1, inplace=True)

    df_av_shap_full = pd.merge(df_av_shap_full, df_av_shap,
                               how='left',
                               on='feature')
    df_av_shap_full.reset_index(inplace=True)

    df_av_shap_full['feature_no_underscore'] = df_av_shap_full['feature'].str.replace("_", " ")

    file_name_4 = file_loc_tools.define_output_file_name(file_loc.loc_shap_averages,
                                                         file_loc.dir_output)
    df_av_shap_full.to_csv(file_loc.dir_output + file_name_4 + '.csv', index=False)

print('finished')
