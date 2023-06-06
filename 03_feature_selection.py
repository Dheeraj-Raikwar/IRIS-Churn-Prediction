'''
Author: Freddie Hampel, JMAN Group
Date: April 2021

Runs after 02_find_feature_correlations.py

Script to:
1. Remove features highly correlated with each other, retaining the one with the highest correlation with churn
2. Select the top 20 features using feature importance in a random forest classifier
'''

import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

from tools.modelling_functions import split_data_into_test_train_wrapper

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools, write_dfs_to_excel

file_loc = FileLocations()
df_feature_matrix = pd.read_pickle(file_loc.loc_feature_matrix_train)

col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

ctrl_remove_correlated = True
dict_export_excel_results = {}

_, _, file_name = file_loc_tools.find_latest_file(file_loc.loc_correlation_test_scores, file_loc.dir_checks)

df_ranked_feature_results = pd.read_excel(file_loc.dir_checks + file_name,
                                          sheet_name='target_var_test_results').sort_values(by='p_value').rename(
    columns={'Unnamed: 0': 'index'})

dict_export_excel_results['full_feature_list'] = df_ranked_feature_results
# remove those with no correlations with churn
condition_no_correlation_with_churn = df_ranked_feature_results['p_value'] < model_config.p_val_thresh_dont_test
df_uncorrelated_with_churn = df_ranked_feature_results[~condition_no_correlation_with_churn].set_index('index')
df_ranked_feature_results = df_ranked_feature_results[condition_no_correlation_with_churn]

print(f'dropped {len(df_uncorrelated_with_churn)} features as below the churn correlation threshold')
dict_export_excel_results['dropped_uncorrelated_with_churn'] = df_uncorrelated_with_churn
dict_export_excel_results['reduced_feature_list_1'] = df_ranked_feature_results

for uncorrelated_feature in df_uncorrelated_with_churn.index:
    print(f'{uncorrelated_feature} not correlated with churn '
          f'({df_uncorrelated_with_churn.loc[uncorrelated_feature, "p_value"]})')

if ctrl_remove_correlated:
    # --- Use statistical test results to narrow down the list of features ------------------------------------------- #

    df_highly_correlated_features = pd.read_excel(file_loc.dir_checks + file_name,
                                                  sheet_name='cross_correlation_test_scores')

    try:
        df_highly_correlated_features = df_highly_correlated_features[
            df_highly_correlated_features['correlation'] >= model_config.remove_corr_features_threshold]
    except TypeError:
        print('stop here')

    df_highly_correlated_features.sort_values(by='correlation', inplace=True)
    df_highly_correlated_features.rename(columns={'Unnamed: 0': 'index'}, inplace=True)
    df_highly_correlated_features.drop_duplicates(subset=['index', 'feature'], inplace=True)

    condition_diagonal_of_corr_matrix = df_highly_correlated_features['index'] != df_highly_correlated_features[
        'feature']
    df_highly_correlated_features = df_highly_correlated_features[condition_diagonal_of_corr_matrix]

    df_correlated_feats_iter = df_highly_correlated_features.copy()
    list_features_uncorrelated = []
    df_features_dropped = pd.DataFrame()

    # create the list of the most important and uncorrelated features
    for feature in df_ranked_feature_results['index']:

        if feature in df_correlated_feats_iter['index'].to_list():
            list_features_uncorrelated.append(feature)

            # find highly correlated features to remove from potential selection list
            list_correlated = df_correlated_feats_iter[df_correlated_feats_iter['index'] == feature][
                'feature'].to_list()

            # print those dropped and add to df for tracking
            for feature_to_drop in list_correlated:
                print(f'dropped {feature_to_drop}, as highly correlated with {feature}')

                score = df_correlated_feats_iter[(df_correlated_feats_iter['index'] == feature) &
                                                 (df_correlated_feats_iter['feature'] == feature_to_drop)][
                    'correlation'].values[0]

                df_features_dropped.loc[feature_to_drop, 'correlated_feature_retained'] = feature
                df_features_dropped.loc[feature_to_drop, 'correlation_score'] = score

            # make the iteration dataframe smaller so we don't add the correlated features
            df_correlated_feats_iter = df_correlated_feats_iter[
                ~df_correlated_feats_iter['index'].isin(list_correlated)]

        elif feature not in df_highly_correlated_features[
            'index'].to_list():  # need this as might not have any correlated
            list_features_uncorrelated.append(feature)

    df_feature_matrix_filtered = df_feature_matrix[
        list_features_uncorrelated + [model_config.target_col, col.parent_id]]

    print(f'{len(list_features_uncorrelated)} features remaining for feature selection')

    # reformat the correlated features so that they can be used for iterative feature selection
    df_features_dropped_for_dict = df_features_dropped.reset_index()[['correlated_feature_retained', 'index']]
    df_features_dropped_for_dict = df_features_dropped_for_dict.groupby('correlated_feature_retained', as_index=False).agg({'index':list})
    dict_dropped_correlated_features = df_features_dropped_for_dict.set_index('correlated_feature_retained').to_dict()
    dict_dropped_correlated_features = dict_dropped_correlated_features['index']

    with open(file_loc.loc_dropped_correlated_features_dict, 'wb') as file:
        pickle.dump(dict_dropped_correlated_features, file)
        file.close()

else:
    list_features = df_ranked_feature_results['index'].to_list()
    df_feature_matrix_filtered = df_feature_matrix[list_features + [model_config.target_col, col.parent_id]]
    df_features_dropped = pd.DataFrame()

dict_export_excel_results['dropped_correlated_features'] = df_features_dropped

# create dataframe with columns for export
df_filtered_columns_2 = pd.DataFrame()
for n, column in enumerate(df_feature_matrix_filtered.columns):
    if column not in col.list_id_vars and column != model_config.target_col:
        df_filtered_columns_2.loc[n, 'feature'] = column

dict_export_excel_results['reduced_feature_list_2'] = df_filtered_columns_2

# --- feature selection modelling ------------------------------------------------------------------------------------ #
df_feature_importance = pd.DataFrame()

for x in tqdm(range(model_config.iterations_feature_selection)):
    df_train_matrix, df_test_matrix = split_data_into_test_train_wrapper(df_feature_matrix_filtered, rand_state=x)

    X_train = df_train_matrix[
        [column for column in df_train_matrix.columns if column not in [model_config.target_col] + col.list_id_vars]]

    y_train = df_train_matrix[model_config.target_col]

    selected_features = SelectFromModel(RandomForestClassifier(n_estimators=100))
    selected_features.fit(X_train, y_train)

    list_selected_features = [*X_train.columns[selected_features.get_support()]]
    list_feature_importances = [*selected_features.estimator_.feature_importances_[selected_features.get_support()]]

    df_feature_importance_new = pd.DataFrame([list_selected_features, list_feature_importances],
                                             index=['feature', f'importance_{x}']).transpose()

    try:
        df_feature_importance = pd.merge(df_feature_importance, df_feature_importance_new,
                                         how='outer',
                                         on='feature')
    except KeyError:
        df_feature_importance = df_feature_importance_new

df_feature_importance.set_index('feature', inplace=True)
df_feature_importance.drop(columns=[column for column in df_feature_importance.columns if 'Unnamed' in column],
                           inplace=True)
df_feature_importance['importance'] = df_feature_importance.mean(axis=1, skipna=True)
df_feature_importance.sort_values(by='importance', ascending=False, inplace=True)
df_feature_importance['proportion_above_average'] = df_feature_importance.count(axis=1).add(-1).divide(
    model_config.iterations_feature_selection)

dict_export_excel_results['feature_importance_results'] = df_feature_importance

list_top_features = df_feature_importance.index.to_list()[0:model_config.top_features_count]

df_feature_matrix_top_features = df_feature_matrix_filtered[
    list_top_features + [model_config.target_col, col.parent_id]]

df_filtered_columns_3 = pd.DataFrame()
for n, column in enumerate(df_feature_matrix_top_features.columns):
    if column not in col.list_id_vars and column != model_config.target_col:
        df_filtered_columns_3.loc[n, 'feature'] = column

dict_export_excel_results['reduced_feature_list_3'] = df_filtered_columns_3

# --- Outputs -------------------------------------------------------------------------------------------------------- #

file_name = file_loc_tools.define_output_file_name(file_loc.loc_feature_selection, file_loc.dir_checks)

write_dfs_to_excel(dict_export_excel_results, file_loc.dir_checks + file_name + '.xlsx')

df_feature_matrix_top_features.to_pickle(file_loc.loc_feature_matrix_top_features)

# for recursive feature selection
df_feature_matrix_filtered.to_pickle(file_loc.loc_feature_matrix_correlations_removed)

print('finished')
