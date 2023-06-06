'''
Author: Freddie Hampel, JMAN Group
Date: April 2021

Runs after alteryx workflows

Script to prepare alteryx output for modelling, including:
1. creating unique index#
2. OHE the categorical variables
3. splitting into the required date ranges
4. dropping any columns with no variation or that have too many categories
5. split into test - train and out of time test
'''

import pandas as pd
import numpy as np
from tqdm import tqdm

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

from tools.modelling_functions import split_data_into_test_train_wrapper

file_loc = FileLocations()
df_feature_matrix = pd.read_csv(file_loc.loc_feature_matrix, encoding='latin')
col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

# --- drop anything before April 2019 due to significantly higher churn before that ---------------------------------- #
df_feature_matrix = df_feature_matrix[df_feature_matrix[col.month] >= model_config.start_train_set_incl]

# create id column and set as index
df_feature_matrix['index'] = ''

for column in col.list_id_vars:
    df_feature_matrix['index'] += df_feature_matrix[column] + '//'

df_feature_matrix.set_index('index', inplace=True)

df_feature_matrix = df_feature_matrix[
    col.list_id_vars_not_cat + col.list_categorical_variables + col.list_numerical_variables
    + [model_config.target_col]]

a = len(df_feature_matrix)
b = len(df_feature_matrix.drop_duplicates(subset=col.list_id_vars))
c = a - b

if c > 0:
    print(f'!! Warning there are {c} duplicates in the dataset, have dropped but needs investigation')
df_feature_matrix.drop_duplicates(subset=col.list_id_vars, inplace=True)

# one hot encode variables that are not already 1/0 flags
for column in tqdm(col.list_categorical_variables_OHE):
    print(column)
    df_feature_matrix = df_feature_matrix.join(
        pd.get_dummies(df_feature_matrix[column], prefix=column))
    if column not in col.list_id_vars:
        df_feature_matrix.drop(columns=column, inplace=True)

for column in tqdm(col.dict_label_encode):
    df_feature_matrix[column] = df_feature_matrix[column].map(col.dict_label_encode[column])

# --- fill nans ------------------------------------------------------------------------------------------------------ #
for column in tqdm(df_feature_matrix.columns):

    # check if column is an OHE column as will not want to drop and won't be in categoricals
    condition_ohe_column = False
    for ohe_column in col.list_categorical_variables_OHE:
        if ohe_column in column:
            condition_ohe_column = True

    if column in col.list_numerical_variables:
        median_val = df_feature_matrix[column].median()
        df_feature_matrix[column].fillna(value=median_val, inplace=True)

        # loop to drop columns where they're not correct in ETL
        if column not in col.list_id_vars:
            try:
                df_feature_matrix[column] = df_feature_matrix[column].astype(float)
            except ValueError:
                print(f'!! Warning, have dropped column {column} as could not convert to float'
                      f', unique values: {df_feature_matrix[column].unique()}')
                df_feature_matrix.drop(columns=column, inplace=True)

    elif column in col.list_categorical_variables or condition_ohe_column:
        try:
            mode_val = df_feature_matrix[column].mode()[0]
            df_feature_matrix[column].fillna(value=mode_val, inplace=True)
        except KeyError:
            print(f'!! Warning, error filling mode for {column}')

        # loop to drop columns where they're not correct in ETL
        if column not in col.list_id_vars:
            try:
                df_feature_matrix[column] = df_feature_matrix[column].astype(int)
            except ValueError:
                print(f'!! Warning, have dropped column {column} as could not convert to int'
                      f', unique values: {df_feature_matrix[column].unique()}')
                df_feature_matrix.drop(columns=column, inplace=True)

    elif column not in col.list_id_vars and column != model_config.target_col:
        print(f'!! Warning, have dropped column {column} as is not in categorical or numerical,'
              f' unique values {df_feature_matrix[column].unique()}')
        df_feature_matrix.drop(columns=column, inplace=True)

# --- remove features where there is no variation in the dataset ----------------------------------------------------- #

for column in df_feature_matrix.columns:
    if df_feature_matrix[column].nunique() == 1:
        print(f'dropped {column} as all values were {df_feature_matrix[column].unique()}')
        df_feature_matrix.drop(columns=column, inplace=True)

# --- split into out of time dataset --------------------------------------------------------------------------------- #
# this is needed as we want a prediction for this most recent row for the output,
# therefore the model should not be trained on this data
list_id_vars_not_month = [column for column in col.list_id_vars if column != col.month]

if model_config.split_data_for_out_of_time_test:
    condition_training_set = df_feature_matrix[col.month] < model_config.date_for_split
    condition_test_out_of_time = ~condition_training_set & \
                                 (df_feature_matrix[col.month] <= model_config.max_test_set_incl)

    df_feature_matrix_out_of_time_test = df_feature_matrix[condition_test_out_of_time]
    df_feature_matrix = df_feature_matrix[condition_training_set]

    df_feature_matrix_out_of_time_test.to_pickle(file_loc.loc_feature_matrix_out_of_time_test)

else:
    df_feature_matrix[f'max_{col.month}'] = df_feature_matrix[col.month].max()

    condition_max_month = df_feature_matrix[f'max_{col.month}'] == df_feature_matrix[col.month]

    df_feature_matrix_max_date = df_feature_matrix[condition_max_month].drop(columns=f'max_{col.month}')
    df_feature_matrix = df_feature_matrix[~condition_max_month].drop(columns=f'max_{col.month}')


# --- split into test and train sets so that test remains unseen ----------------------------------------------------- #

df_test, df_train = split_data_into_test_train_wrapper(df_feature_matrix)

# -- output data ----------------------------------------------------------------------------------------------------- #

df_train.to_pickle(file_loc.loc_feature_matrix_train)
df_test.to_pickle(file_loc.loc_feature_matrix_test)

print('finished')
