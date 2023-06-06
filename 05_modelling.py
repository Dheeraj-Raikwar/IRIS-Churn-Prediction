'''
Author: Freddie Hampel, JMAN Group
Date: April 2021

Runs after 01_feature_matrix_prep.py,
or after 03_feature_selection.py if building a baseline model,
or after 04_iterative_feature_selection_substitution.py if iterative feature selection ahs been used.
In this case, the feature list in the config needs to be updated to be in line with the desired features


Script to:
1. Split the data into test train and validation sets
2. Tune model hyperparameters (optional)
3. Train model and produce accuracy metrics
4. Calibrate model to produce representative probability predictions
'''

from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import shap
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.calibration import _CalibratedClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

from tools.modelling_functions import perform_bayesian_optimisation, split_data_into_test_train_wrapper

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

file_loc = FileLocations()
df_feature_matrix_train = pd.read_pickle(file_loc.loc_feature_matrix_train)
df_test = pd.read_pickle(file_loc.loc_feature_matrix_test)
df_out_of_time_test = pd.read_pickle(file_loc.loc_feature_matrix_out_of_time_test)

col = ColumnHeaders(df_feature_matrix_train)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

# --- split into test-train ------------------------------------------------------------------------------------------ #

df_feature_matrix_train = df_feature_matrix_train[
    model_config.latest_feature_list + [col.parent_id, model_config.target_col]]
df_test = df_test[model_config.latest_feature_list + [col.parent_id, model_config.target_col]]
df_out_of_time_test = df_out_of_time_test[model_config.latest_feature_list + [col.parent_id, model_config.target_col]]

df_val, df_train = split_data_into_test_train_wrapper(df_feature_matrix_train)

list_columns = [column for column in df_test if column not in [col.parent_id, model_config.target_col]]

X_train = df_train[list_columns]
X_test = df_test[list_columns]
X_val = df_val[list_columns]
X_out_of_time_test = df_out_of_time_test[list_columns]

y_train = df_train[model_config.target_col]
y_test = df_test[model_config.target_col]
y_val = df_val[model_config.target_col]
y_out_of_time_test = df_out_of_time_test[model_config.target_col]

if model_config.ctrl_tune_hyperparams:
    dict_hyp_space, list_int_params = model_config.define_hyperparameters_space(model_config.model_name)
    model, dict_best_params = perform_bayesian_optimisation(model_config.bayes_evals,
                                                            file_loc.dir_models,
                                                            dict_hyp_space,
                                                            model_config.model(),
                                                            list_int_params,
                                                            df_train,
                                                            model_config.model_name,
                                                            model_config.target_col)

    with open(file_loc.loc_model_hyperparams, 'wb') as file:
        pickle.dump(dict_best_params, file)
        file.close()


else:

    model = CatBoostClassifier(iterations=500,
                               verbose=False,
                               eval_metric='AUC',
                               class_weights=model_config.class_weights,
                               # logloss for 2-class classification
                               loss_function='Logloss',
                               # built in facility to prevent overfitting
                               use_best_model=True)

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # --- accuracy metrics ------------------------------------------------------------------------------------------- #

roc_auc_score_train = roc_auc_score(y_train, model.predict(X_train))
roc_auc_score_val = roc_auc_score(y_val, model.predict(X_val))
roc_auc_score_test = roc_auc_score(y_test, model.predict(X_test))
roc_auc_score_out_of_time_test = roc_auc_score(y_out_of_time_test, model.predict(X_out_of_time_test))

print(f'ROC-AUC scores:'
      f'\nTrain: {roc_auc_score_train}'
      f'\nVal: {roc_auc_score_val}'
      f'\nTest: {roc_auc_score_test}'
      f'\nOut of time test: {roc_auc_score_out_of_time_test}')

if model_config.calibrate_probabilities:
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(model.predict_proba(X_train)[:, 1], y_train)

    calibrated_model = _CalibratedClassifier(model, [calibrator], method='isotonic', classes=model.classes_)

    roc_auc_score_train_cal = roc_auc_score(y_train, calibrated_model.predict_proba(X_train)[:, 1])
    roc_auc_score_val_cal = roc_auc_score(y_val, calibrated_model.predict_proba(X_val)[:, 1])
    roc_auc_score_test_cal = roc_auc_score(y_test, calibrated_model.predict_proba(X_test)[:, 1])
    roc_auc_score_out_of_time_test_cal = roc_auc_score(y_out_of_time_test,
                                                       calibrated_model.predict_proba(X_out_of_time_test)[:, 1])

    print(f'ROC-AUC scores calibrated:'
          f'\nTrain: {roc_auc_score_train_cal}'
          f'\nVal: {roc_auc_score_val_cal}'
          f'\nTest: {roc_auc_score_test_cal}'
          f'\nOut of time test: {roc_auc_score_out_of_time_test_cal}')

    with open(file_loc.loc_model_calibrated, 'wb') as file:
        pickle.dump(calibrated_model, file)
        file.close()

feature_importance = pd.DataFrame(model.feature_importances_,
                                  index=X_train.columns,
                                  columns=['importance']).sort_values(by='importance', ascending=False)

with open(file_loc.loc_model, 'wb') as file:
    pickle.dump(model, file)
    file.close()

print('finished')
