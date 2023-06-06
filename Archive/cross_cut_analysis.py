import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression
from scipy.stats import chi2_contingency, pearsonr, f_oneway, pointbiserialr

from tools.tools import write_dfs_to_excel, FileLocationTools
from config.config import FileLocations, ColumnHeaders, ModelConfig

# --- import data ---------------------------------------------------------------------------------------------------- #

file_loc = FileLocations()

df_feature_matrix = pd.read_pickle(file_loc.loc_feature_matrix_train)

file_loc_tools = FileLocationTools()
col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()

# --- Total customer ARR --------------------------------------------------------------------------------------------- #

list_max_values_arr = [5000, 10000, 15000, 20000, 25000, 1000000]

condition_ARR_group_1 = df_feature_matrix['Total_Monthly_Customer_ARR'] < list_max_values_arr[0]

condition_ARR_group_2 = df_feature_matrix['Total_Monthly_Customer_ARR'] < list_max_values_arr[
    1]  # to be used in sequence

condition_ARR_group_3 = df_feature_matrix['Total_Monthly_Customer_ARR'] < list_max_values_arr[2]

condition_ARR_group_4 = df_feature_matrix['Total_Monthly_Customer_ARR'] < list_max_values_arr[3]

condition_ARR_group_5 = df_feature_matrix['Total_Monthly_Customer_ARR'] < list_max_values_arr[4]

condition_ARR_group_6 = df_feature_matrix['Total_Monthly_Customer_ARR'] < list_max_values_arr[5]

list_arr_conditions = [condition_ARR_group_1, condition_ARR_group_2, condition_ARR_group_3, condition_ARR_group_4,
                       condition_ARR_group_5, condition_ARR_group_6]

df_feature_matrix_filtered = df_feature_matrix[df_feature_matrix['Total_Monthly_Customer_ARR'] >= 0]
df_results_arr = pd.DataFrame()

for n, condition in enumerate(list_arr_conditions):
    df_temp = df_feature_matrix_filtered[condition]

    df_results_arr.loc[n, 'max_ARR'] = df_temp["Total_Monthly_Customer_ARR"].max()
    df_results_arr.loc[n, 'customers'] = df_temp[col.parent_id].nunique()
    df_results_arr.loc[n, 'churn_prob'] = df_temp[model_config.target_col].mean()

    df_feature_matrix_filtered = df_feature_matrix_filtered[~condition]

# --- interactions --------------------------------------------------------------------------------------------------- #
list_max_values_interactions = [1, 5, 10, 15, 20, 100]

# below conditions to be used in sequence
condition_interactions_group_1 = df_feature_matrix['LTM_interactions'] == list_max_values_interactions[0]

condition_interactions_group_2 = df_feature_matrix['LTM_interactions'] < list_max_values_interactions[1]

condition_interactions_group_3 = df_feature_matrix['LTM_interactions'] < list_max_values_interactions[2]

condition_interactions_group_4 = df_feature_matrix['LTM_interactions'] < list_max_values_interactions[3]

condition_interactions_group_5 = df_feature_matrix['LTM_interactions'] < list_max_values_interactions[4]

condition_interactions_group_6 = df_feature_matrix['LTM_interactions'] < list_max_values_interactions[5]

list_interactions_conditions = [condition_interactions_group_1, condition_interactions_group_2,
                                condition_interactions_group_3, condition_interactions_group_4,
                                condition_interactions_group_5, condition_interactions_group_6]

df_feature_matrix_filtered_2 = df_feature_matrix[df_feature_matrix['Total_Monthly_Customer_ARR'] >= 0]
df_results_arr_interactions = pd.DataFrame()
n = 0

for arr_counter, condition_arr in enumerate(list_arr_conditions):
    for interactions_counter, condition_interactions in enumerate(list_interactions_conditions):
        df_temp = df_feature_matrix_filtered_2[condition_arr & condition_interactions]

        df_results_arr_interactions.loc[n, 'max_ARR'] = list_max_values_arr[arr_counter]
        df_results_arr_interactions.loc[n, 'max_interactions'] = list_max_values_interactions[interactions_counter]
        df_results_arr_interactions.loc[n, 'customers'] = df_temp[col.parent_id].nunique()
        df_results_arr_interactions.loc[n, 'churn_prob'] = df_temp[model_config.target_col].mean()

        df_feature_matrix_filtered_2 = df_feature_matrix_filtered_2[~(condition_arr & condition_interactions)]
        n += 1

df_output_pivot = df_results_arr_interactions.pivot(index='max_interactions', columns='max_ARR', values='churn_prob')

print('script finished')
