'''
Author: Freddie Hampel, JMAN Group
Date: April 2021

Runs after 01_feature_matrix_prep.py

Script to find correlations between:
1. Churn and the potential features - removing any that are not correlated with churn
2. Feature cross-correlations - flagging features correlated with each other, to be removed in 03_feature_selection.py
'''

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

dict_test_results = {}
ctrl_run_cross_correlations = True

df_feature_matrix.drop_duplicates(subset=col.parent_id, keep='last', inplace=True)


# --- functions for this script -------------------------------------------------------------------------------------- #

def create_df_for_correlation(scores, p_values, test_feature, list_features):
    df_feature_correlation_matrix_new = pd.DataFrame([scores, p_values],
                                                     columns=list_features,
                                                     index=[test_feature, f'{test_feature}_p_val']).transpose()

    df_feature_correlation_matrix_new.reset_index(inplace=True)

    return df_feature_correlation_matrix_new


def cramers_v(confusion_matrix, chi2):
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def run_chi2_test(X, y, df_results, use_cramers_v=True):
    Xy = pd.concat([X, y], axis=1)
    Xy['dummy'] = 0

    var_1 = [*Xy.columns][0]
    var_2 = [*Xy.columns][1]

    Xy_grouped = Xy.groupby([*Xy.columns], as_index=False)['dummy'].count()

    Xy_grouped = Xy_grouped.pivot(index=var_1, columns=var_2, values='dummy').fillna(0)

    chi2, p, dof, expected = chi2_contingency(Xy_grouped)

    if use_cramers_v:
        correlation = cramers_v(Xy_grouped, chi2)
        df_results.loc[var_1, 'correlation'] = correlation

    df_results.loc[var_1, 'chi2_score'] = chi2
    df_results.loc[var_1, 'p_value'] = p

    return df_results


# --- Chi2 test ------------------------------------------------------------------------------------------------------ #

# keep only categorical variables for the chi-squared test
# keep parent id so that we can split customers into test-train
df_feature_matrix_categorical = df_feature_matrix[
    col.list_categorical_variables + [col.parent_id, model_config.target_col]]

y = df_feature_matrix_categorical[model_config.target_col]

X_categorical = df_feature_matrix_categorical.drop(columns=[col.parent_id,
                                                            col.product_family,
                                                            model_config.target_col])

# use SelectKBest to perform chi2 test
select_k_best_model = SelectKBest(score_func=chi2, k='all')
select_k_best_model.fit(X_categorical, y)

chi2_pvalues = select_k_best_model.pvalues_
chi2_scores = select_k_best_model.scores_

df_categorical_feature_testing_results = pd.DataFrame([chi2_scores, chi2_pvalues],
                                                      columns=X_categorical.columns,
                                                      index=['chi2_score', 'p_value']).transpose()

# filter out the insignificant results
df_categorical_feature_testing_results.sort_values(by='chi2_score', ascending=False, inplace=True)

# --- ANOVA F measure for numerical input, categorical output -------------------------------------------------------- #

X_numerical = df_feature_matrix[col.list_numerical_variables]

anova_model = SelectKBest(score_func=f_classif, k='all')
anova_model.fit(X_numerical, y)

anova_scores = anova_model.scores_
anova_p_values = anova_model.pvalues_

df_numerical_feature_testing_results = pd.DataFrame([anova_scores, anova_p_values],
                                                    columns=X_numerical.columns,
                                                    index=['anova_score', 'p_value']).transpose()

df_numerical_feature_testing_results.sort_values(by='anova_score', inplace=True, ascending=False)

# --- join test scores for export ------------------------------------------------------------------------------------ #

df_numerical_feature_testing_results.rename(columns={'anova_score': 'test_score'}, inplace=True)
df_numerical_feature_testing_results['test'] = "ANOVA"

df_categorical_feature_testing_results.rename(columns={'chi2_score': 'test_score'}, inplace=True)
df_categorical_feature_testing_results['test'] = "Chi2"

df_target_var_test_scores = df_numerical_feature_testing_results.append(df_categorical_feature_testing_results)

dict_test_results['target_var_test_results'] = df_target_var_test_scores.copy()

# --- get features above the threshold for correlation --------------------------------------------------------------- #

target_var_test_scores_significant = df_target_var_test_scores[
    df_target_var_test_scores['p_value'] <= model_config.p_val_threshold]

dict_test_results['target_var_significant_features'] = target_var_test_scores_significant

# --- cross-feature correlations ------------------------------------------------------------------------------------- #

if ctrl_run_cross_correlations:

    df_feature_correlation_matrix_chi2 = pd.DataFrame()
    df_feature_correlation_matrix_anova = pd.DataFrame()
    df_feature_correlation_matrix_num = pd.DataFrame()

    for categorical_feature in tqdm(X_categorical.columns):

        y = X_categorical[categorical_feature]

        # categorical and categorical
        list_features = [column for column in X_categorical.columns if column != categorical_feature]
        X = X_categorical[list_features]

        for feature in list_features:
            df_feature_correlation_matrix_chi2 = run_chi2_test(X[feature],
                                                               y,
                                                               df_feature_correlation_matrix_chi2)

        df_feature_correlation_matrix_chi2.rename(columns={'correlation': categorical_feature}, inplace=True)
        df_feature_correlation_matrix_chi2.drop(columns=['p_value', 'chi2_score'], inplace=True)

        # categorical and numerical
        list_features = [column for column in X_numerical]
        X = X_numerical[list_features]
        for feature in list_features:
            corr, _ = pointbiserialr(X[feature], y)
            df_feature_correlation_matrix_anova.loc[feature, categorical_feature] = abs(corr)

    for numerical_feature in tqdm(X_numerical.columns):
        y = X_numerical[numerical_feature]

        # numerical and categorical
        list_features = [column for column in X_categorical.columns]
        X = X_categorical[list_features]
        for feature in list_features:
            corr, _ = pointbiserialr(X[feature], y)
            df_feature_correlation_matrix_anova.loc[feature, numerical_feature] = abs(corr)

    # numerical and numerical
    df_feature_correlation_matrix_num = X_numerical.corr()

    for column in df_feature_correlation_matrix_num.columns:
        df_feature_correlation_matrix_num[column] = df_feature_correlation_matrix_num[column].abs()

    df_results_cross_correlation = pd.DataFrame()

    # melt and compile results
    df_feature_correlation_matrix_chi2 = df_feature_correlation_matrix_chi2.melt(value_name='correlation',
                                                                                 ignore_index=False,
                                                                                 var_name='feature')

    df_feature_correlation_matrix_anova = df_feature_correlation_matrix_anova.melt(value_name='correlation',
                                                                                   ignore_index=False,
                                                                                   var_name='feature')

    df_feature_correlation_matrix_num = df_feature_correlation_matrix_num.melt(value_name='correlation',
                                                                               ignore_index=False,
                                                                               var_name='feature')
    df_feature_correlation_matrix_num['test'] = "Pearson's correlation"
    df_feature_correlation_matrix_chi2['test'] = "Cramer's V"
    df_feature_correlation_matrix_anova['test'] = "Point Biserial"

    df_results_cross_correlation = df_feature_correlation_matrix_chi2.append(df_feature_correlation_matrix_num).append(
        df_feature_correlation_matrix_anova)

    df_results_cross_correlation = df_results_cross_correlation[
        df_results_cross_correlation['correlation'] == df_results_cross_correlation['correlation']]

    dict_test_results['cross_correlation_test_scores'] = df_results_cross_correlation

    # --- significant correlation test scores ------------------------------------------------------------------------ #

file_name = file_loc_tools.define_output_file_name(file_loc.loc_correlation_test_scores, file_loc.dir_checks)

write_dfs_to_excel(dict_test_results, file_loc.dir_checks + file_name + '.xlsx')

print('finished')
