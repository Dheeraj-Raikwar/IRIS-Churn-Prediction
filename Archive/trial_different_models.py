from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import shap
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

from tools.modelling_functions import perform_bayesian_optimisation, split_data_into_test_train_wrapper

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

file_loc = FileLocations()
df_feature_matrix = pd.read_pickle(file_loc.loc_feature_matrix_top_features)

col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

# --- split into test-train ------------------------------------------------------------------------------------------ #

df_test, df_train = split_data_into_test_train_wrapper(df_feature_matrix)
df_val, df_train = split_data_into_test_train_wrapper(df_train)

X_train = df_train[[column for column in df_train if column not in [col.parent_id, model_config.target_col]]]
X_test = df_test[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]
X_val = df_val[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]

y_train = df_train[model_config.target_col]
y_test = df_test[model_config.target_col]
y_val = df_val[model_config.target_col]

dict_hyp_space_catboost, list_int_params_catboost = model_config.define_hyperparameters_space('CatBoost')
catboost_model, dict_best_catboost_params = perform_bayesian_optimisation(model_config.bayes_evals,
                                                                          file_loc.dir_models,
                                                                          dict_hyp_space_catboost,
                                                                          CatBoostClassifier(),
                                                                          list_int_params_catboost,
                                                                          df_train,
                                                                          'CatBoost',
                                                                          model_config.target_col)

dict_hyp_space_lightgbm, list_int_params_lightgbm = model_config.define_hyperparameters_space('LightGBM')
lightgbm, dict_best_lightgbm_params = perform_bayesian_optimisation(model_config.bayes_evals,
                                                                    file_loc.dir_models,
                                                                    dict_hyp_space_lightgbm,
                                                                    LGBMClassifier(),
                                                                    list_int_params_lightgbm,
                                                                    df_train,
                                                                    'LightGBM',
                                                                    model_config.target_col)

dict_hyp_space_xgboost, list_int_params_xgboost = model_config.define_hyperparameters_space('XGBoost')
xgboost, dict_best_xgboost_params = perform_bayesian_optimisation(model_config.bayes_evals,
                                                                  file_loc.dir_models,
                                                                  dict_hyp_space_xgboost,
                                                                  XGBClassifier(),
                                                                  list_int_params_xgboost,
                                                                  df_train,
                                                                  'XGBoost',
                                                                  model_config.target_col)

xgboost_rf, dict_best_xgboost_rf_params = perform_bayesian_optimisation(model_config.bayes_evals,
                                                                        file_loc.dir_models,
                                                                        dict_hyp_space_xgboost,
                                                                        XGBRFClassifier(),
                                                                        list_int_params_xgboost,
                                                                        df_train,
                                                                        'XGBoostRF',
                                                                        model_config.target_col)

dict_best_params = {'XGBoostRF': dict_best_xgboost_rf_params,
                    'XGBoost': dict_best_xgboost_params,
                    'CatBoost': dict_best_catboost_params,
                    'LightGBM': dict_best_lightgbm_params}

with open(file_loc.loc_multiple_model_hyperparams, 'wb') as file:
    pickle.dump(dict_best_params, file)
    file.close()

print('finished')