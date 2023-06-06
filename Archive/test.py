import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from tools.modelling_functions import split_data_set_into_test_train
from tools.tools import write_dfs_to_excel, FileLocationTools
from config.config import FileLocations, ColumnHeaders, ModelConfig

# --- import data ---------------------------------------------------------------------------------------------------- #

file_loc = FileLocations()

df_feature_matrix = pd.read_pickle(file_loc.loc_feature_matrix_train)

file_loc_tools = FileLocationTools()
col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()

dict_test_results = {}

df_results_cross_correlation = pd.DataFrame()

_, _, file_name = file_loc_tools.find_latest_file(file_loc.loc_correlation_test_scores, file_loc.dir_checks)
df_feature_correlation_matrix_anova = pd.read_excel(file_loc.dir_checks + file_name,
                                              sheet_name='cross_feat_anova',
                                              index_col='Unnamed: 0')

df_feature_correlation_matrix_num = pd.read_excel(file_loc.dir_checks + file_name,
                                              sheet_name='cross_feat_num',
                                              index_col='Unnamed: 0')
df_feature_correlation_matrix_chi2 = pd.read_excel(file_loc.dir_checks + file_name,
                                              sheet_name='cross_feat_chi2',
                                              index_col='Unnamed: 0')

X_numerical = df_feature_matrix[col.list_numerical_variables]
X_categorical = df_feature_matrix[col.list_categorical_variables]

for feature in tqdm(df_feature_matrix.columns):

    feature_bool = False

    if feature in X_numerical.columns:
        # conditions to get rid of nans and ensure we keep the correct correlation
        condition_keep_row_anova = df_feature_correlation_matrix_anova[f'{feature}_p_val'] == \
                                   df_feature_correlation_matrix_anova[f'{feature}_p_val']
        condition_keep_row_num = df_feature_correlation_matrix_num[f'{feature}_p_val'] == \
                                 df_feature_correlation_matrix_num[f'{feature}_p_val']

        # append so there's a single column for this feature
        df_feature = df_feature_correlation_matrix_anova[condition_keep_row_anova][
            ['index', f'{feature}_p_val']].append(
            df_feature_correlation_matrix_num[condition_keep_row_num][['index', f'{feature}_p_val']])

        feature_bool = True

    elif feature in X_categorical.columns:
        # conditions to get rid of nans and ensure we keep the correct correlation
        condition_keep_row_anova = df_feature_correlation_matrix_anova[f'{feature}_p_val'] == \
                                   df_feature_correlation_matrix_anova[f'{feature}_p_val']
        condition_keep_row_chi2 = df_feature_correlation_matrix_chi2[f'{feature}_p_val'] == \
                                  df_feature_correlation_matrix_chi2[f'{feature}_p_val']

        # append so there's a single column for this feature
        df_feature = df_feature_correlation_matrix_anova[condition_keep_row_anova][['index', f'{feature}_p_val']].append(
            df_feature_correlation_matrix_chi2[condition_keep_row_chi2][['index', f'{feature}_p_val']])

        feature_bool = True

    elif feature not in col.list_id_vars and feature != model_config.target_col:
        print(f'!!! Warning, could not find {feature} in columns')

    if feature_bool:
        try:
            df_results_cross_correlation = pd.merge(df_results_cross_correlation,
                                                    df_feature,
                                                    how='outer',
                                                    on='index')
        except KeyError:
            df_results_cross_correlation = df_feature

dict_test_results['cross_correlation_test_scores'] = df_results_cross_correlation


dict_test_results['cross_feat_chi2'] = df_feature_correlation_matrix_chi2
dict_test_results['cross_feat_anova'] = df_feature_correlation_matrix_anova
dict_test_results['cross_feat_num'] = df_feature_correlation_matrix_num
# --- significant correlation test scores ------------------------------------------------------------------------ #

df_target_var_test_scores = pd.read_excel(file_loc.dir_checks + file_name,
                                              sheet_name='target_var_test_results',
                                              index_col='Unnamed: 0')

dict_test_results['target_var_test_results'] = df_target_var_test_scores

# --- get features above the threshold for correlation --------------------------------------------------------------- #
target_var_test_scores_significant = pd.read_excel(file_loc.dir_checks + file_name,
                                              sheet_name='target_var_significant_features',
                                              index_col='Unnamed: 0')

dict_test_results['target_var_significant_features'] = target_var_test_scores_significant

file_name = file_loc_tools.define_output_file_name(file_loc.loc_correlation_test_scores, file_loc.dir_checks)

write_dfs_to_excel(dict_test_results, file_loc.dir_checks + file_name + '.xlsx')