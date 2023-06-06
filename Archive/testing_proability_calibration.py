from catboost import CatBoostClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier
import shap
import pickle

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import roc_auc_score
from sklearn.calibration import _CalibratedClassifier
from sklearn.isotonic import IsotonicRegression

from tools.modelling_functions import perform_bayesian_optimisation, split_data_into_test_train_wrapper

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

file_loc = FileLocations()
df_feature_matrix = pd.read_pickle(file_loc.loc_feature_matrix_train)

col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

# --- split into test-train ------------------------------------------------------------------------------------------ #

df_feature_matrix = df_feature_matrix[model_config.latest_feature_list + [ col.parent_id, model_config.target_col]]

df_test, df_train = split_data_into_test_train_wrapper(df_feature_matrix)
df_val, df_train = split_data_into_test_train_wrapper(df_train)

X_train = df_train[[column for column in df_train if column not in [col.parent_id, model_config.target_col]]]
X_test = df_test[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]
X_val = df_val[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]

y_train = df_train[model_config.target_col]
y_test = df_test[model_config.target_col]
y_val = df_val[model_config.target_col]

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
    # with open(file_loc.loc_model_hyperparams, 'rb') as file:
    #     dict_best_params = pickle.load(file)
    #     file.close()
    #
    # model = CatBoostClassifier(**dict_best_params['CatBoost'])

    model = CatBoostClassifier(iterations=500,
                               verbose=False,
                               eval_metric='AUC',
                               class_weights={0: 1,
                                              1: 12},  # model_config.class_weights,
                               # logloss for 2-class classification
                               loss_function='Logloss',
                               # built in facility to prevent overfitting
                               use_best_model=True)

    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # --- accuracy metrics ------------------------------------------------------------------------------------------- #

    roc_auc_score_test = roc_auc_score(y_test, model.predict(X_test))
    roc_auc_score_train = roc_auc_score(y_train, model.predict(X_train))

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(model.predict_proba(X_train)[:, 1], y_train)
    # wrapper gets around bug in scikit-learn new version
    calibrated_model = _CalibratedClassifier(model, [calibrator], method='isotonic', classes=model.classes_)

    roc_auc_score_test_cal = roc_auc_score(y_test, calibrated_model.predict_proba(X_test)[:, 1])
    roc_auc_score_train_cal = roc_auc_score(y_train, calibrated_model.predict_proba(X_train)[:, 1])

feature_importance = pd.DataFrame(model.feature_importances_,
                                  index=X_train.columns,
                                  columns=['importance']).sort_values(by='importance', ascending=False)

with open(file_loc.loc_model, 'wb') as file:
    pickle.dump(model, file)
    file.close()

with open(file_loc.loc_model_calibrated, 'wb') as file:
    pickle.dump(calibrated_model, file)
    file.close()

# --- shapley values ------------------------------------------------------------------------------------------------- #

print('finished')
