from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import shap
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score, roc_auc_score

from tools.modelling_functions import perform_bayesian_optimisation, split_data_into_test_train_wrapper

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

file_loc = FileLocations()
df_feature_matrix = pd.read_pickle(file_loc.loc_feature_matrix_correlations_removed)

col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

# --- sequential feature selection ----------------------------------------------------------------------------------- #
X = df_feature_matrix[
    [column for column in df_feature_matrix.columns if column not in [model_config.target_col] + col.list_id_vars]]
y = df_feature_matrix[model_config.target_col]

df_test, df_train = split_data_into_test_train_wrapper(df_feature_matrix)
df_val, df_train = split_data_into_test_train_wrapper(df_train)

X_train = df_train[[column for column in df_train if column not in [col.parent_id, model_config.target_col]]]
X_test = df_test[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]
X_val = df_val[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]

y_train = df_train[model_config.target_col]
y_test = df_test[model_config.target_col]
y_val = df_val[model_config.target_col]

dict_results = {}

for x in tqdm(range(1,68)):
    iterative_feature_forwards = SequentialFeatureSelector(CatBoostClassifier(),
                                                           n_features_to_select=x,
                                                           scoring='roc_auc',
                                                           n_jobs=-1,
                                                           cv=3)
    iterative_feature_forwards.fit(X, y)

    X_train_forwards = iterative_feature_forwards.transform(X_train)
    X_val_forwards = iterative_feature_forwards.transform(X_val)
    X_test_forwards = iterative_feature_forwards.transform(X_test)

    model = CatBoostClassifier(iterations=500,
                               verbose=True,
                               eval_metric='AUC',
                               class_weights=model_config.class_weights,
                               # logloss for 2-class classification
                               loss_function='Logloss',
                               # built in facility to prevent overfitting
                               use_best_model=True)

    model.fit(X_train_forwards, y_train, eval_set=(X_val_forwards, y_val))
    score = roc_auc_score(y_test, model.predict(X_test_forwards))

    dict_results[f'forwards_{x}'] = {'features': x,
                                     'direction': 'forwards',
                                     'list_features': [*X_val.columns],
                                     'roc_auc_test': score}

    iterative_feature_backwards = SequentialFeatureSelector(CatBoostClassifier(),
                                                            n_features_to_select=x,
                                                            direction='backward',
                                                            scoring='roc_auc',
                                                            n_jobs=-1,
                                                            cv=3)
    iterative_feature_backwards.fit(X, y)

    X_train_backwards = iterative_feature_backwards.transform(X_train)
    X_val_backwards = iterative_feature_backwards.transform(X_val)
    X_test_backwards = iterative_feature_backwards.transform(X_test)

    model = CatBoostClassifier(iterations=500,
                               verbose=True,
                               eval_metric='AUC',
                               class_weights=model_config.class_weights,
                               # logloss for 2-class classification
                               loss_function='Logloss',
                               # built in facility to prevent overfitting
                               use_best_model=True)

    model.fit(X_train_backwards, y_train, eval_set=(X_val_backwards, y_val))
    score = roc_auc_score(y_test, model.predict(X_test_backwards))

    dict_results[f'backwards_{x}'] = {'features': x,
                                      'direction': 'backwards',
                                      'list_features': [*X_val.columns],
                                      'roc_auc_test': score}

    with open(file_loc.loc_model_iterations, 'wb') as file:
        pickle.dump(dict_results, file)
        file.close()

# --- substitute correlated features --------------------------------------------------------------------------------- #

# --- take 2 sequential feature selection ---------------------------------------------------------------------------- #

print('finished')
