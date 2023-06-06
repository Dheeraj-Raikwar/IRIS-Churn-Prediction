"""
Author: Freddie Hampel (JMAN Group)
Script: modelling_functions
Date: April 2021

This script handles the hyperparameter tuning used oin train_predictive_model through the main
func_bayesian_optimisation which performs bayesian optimisation on the ML model

Script is not run independently
"""

import csv
from timeit import default_timer as timer
import time
import numpy as np
import pandas as pd
from math import exp
import shap

from hyperopt import STATUS_OK, hp, tpe, Trials, fmin
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from catboost import CatBoostClassifier
import catboost

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

file_loc = FileLocations()
col = ColumnHeaders()
model_config = ModelConfig()
file_tools = FileLocationTools()


def split_data_into_test_train_wrapper(df_feature_matrix,
                                       unique_variable=col.parent_id,
                                       test_split=model_config.test_split,
                                       rand_state=model_config.rand_state,
                                       split_target_evenly=True):
    def split_data_set_into_test_train(df_feature_matrix=df_feature_matrix,
                                       unique_variable=unique_variable,
                                       test_split=test_split,
                                       rand_state=rand_state):
        """
        Split the dataset into a test-train set based on a specific unique variable.
        This is needed to avoid falsely inflating the accuracy if e.g. there are several observations for the same customer
        :param df_feature_matrix:
        :param target_variable:
        :param unique_variable:
        :param test_split:
        :return:
        """
        df_unique = df_feature_matrix.drop_duplicates(subset=unique_variable)

        df_train_unique, df_test_unique = train_test_split(df_unique, test_size=test_split, random_state=rand_state)

        list_train_unique = df_train_unique[unique_variable].to_list()
        list_test_unique = df_test_unique[unique_variable].to_list()

        df_features_train = df_feature_matrix[
            df_feature_matrix[unique_variable].isin(list_train_unique)]
        df_features_test = df_feature_matrix[df_feature_matrix[unique_variable].isin(list_test_unique)]

        return df_features_train, df_features_test

    if split_target_evenly:
        # need to keep churn split fairly similar across test and train to avoid skew
        # make sure a customer is still only in one set
        # df_feature_matrix_churn = df_feature_matrix[df_feature_matrix[model_config.target_col] == 1]
        # df_feature_matrix_not_churn = df_feature_matrix[df_feature_matrix[model_config.target_col] == 0]

        list_churned_customers = [*df_feature_matrix[df_feature_matrix[model_config.target_col] == 1][
            unique_variable].unique()]

        df_feature_matrix_churn = df_feature_matrix[df_feature_matrix[unique_variable].isin(list_churned_customers)]
        df_feature_matrix_not_churn = df_feature_matrix[
            ~df_feature_matrix[unique_variable].isin(list_churned_customers)]

        df_train_churn, df_test_churn = split_data_set_into_test_train(df_feature_matrix_churn)
        df_train_not_churn, df_test_not_churn = split_data_set_into_test_train(df_feature_matrix_not_churn)

        df_train = df_train_churn.append(df_train_not_churn)
        df_test = df_test_churn.append(df_test_not_churn)
    else:
        df_train, df_test = split_data_set_into_test_train(df_feature_matrix)

    return df_test, df_train


def func_objective_wrapper(output_dir, list_int_params, algorithm, train_set, target_name, name,
                           unique_column, validation_set=True):
    '''
    Need to write wrapper function to add defaults into objective function
    :param output_dir:
    :param list_int_params:
    :param model:
    :param train_set:
    :param n_folds:
    :return:
    '''
    if validation_set:
        def func_loss(dict_params, alg=algorithm, train_set=train_set, target_name=target_name,
                      unique_column=unique_column, rand_state=model_config.rand_state):

            df_features_train_test, df_features_val = split_data_into_test_train_wrapper(train_set, unique_column,
                                                                                         rand_state=rand_state)
            df_features_train, df_features_test = split_data_into_test_train_wrapper(df_features_train_test,
                                                                                     unique_column,
                                                                                     rand_state=rand_state)

            X_train = df_features_train.drop(columns=[target_name, unique_column])
            X_test = df_features_test.drop(columns=[target_name, unique_column])
            X_val = df_features_val.drop(columns=[target_name, unique_column])

            y_train = df_features_train[target_name]
            y_test = df_features_test[target_name]
            y_val = df_features_val[target_name]

            # fit model to train
            try:
                model = alg.set_params(**dict_params)
            except catboost.CatBoostError:
                model = CatBoostClassifier(**dict_params)
            try:
                model_trained = model.fit(X_train, y_train, eval_set=(X_val, y_val))
            except ValueError:
                model_trained = model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            # predict test set
            predictions = model_trained.predict(X_test)

            # using test roc auc as loss function
            roc_auc = roc_auc_score(y_test, predictions)
            # 1 - roc_auc for a loss function that can be minimised
            loss = 1 - roc_auc

            return loss
    else:
        def func_loss(dict_params, alg=algorithm, train_set=train_set, target_name=target_name,
                      unique_column=unique_column, n_folds=model_config.n_folds):

            loss_sum = 0

            for k in range(n_folds):
                # use different splits to make sure it's not a based on the split
                # test-train split this needs to be done within the train set for tuning
                # need a validation set to use built in overfitting prevention
                df_features_train, df_features_test = split_data_into_test_train_wrapper(train_set,
                                                                                         unique_column,
                                                                                         rand_state=k)

                X_train = df_features_train.drop(columns=[target_name, unique_column])
                X_test = df_features_test.drop(columns=[target_name, unique_column])

                y_train = df_features_train[target_name]
                y_test = df_features_test[target_name]

                # fit model to train
                try:
                    model = alg.set_params(**dict_params)
                except catboost.CatBoostError:
                    model = CatBoostClassifier(**dict_params)

                model_trained = model.fit(X_train, y_train)

                # predict test set
                predictions = model_trained.predict(X_test)

                # using test roc auc as loss function
                roc_auc = roc_auc_score(y_test, predictions)
                # 1 - roc_auc for a loss function that can be minimised
                loss = 1 - roc_auc

                loss_sum += loss

            return loss_sum / n_folds

    def func_objective(dict_params, output_dir=output_dir, list_int_params=list_int_params, name=name):
        '''
        Objective function for GBM hyperparameter optimisation
        Note - algorithm must have a cv function
        :param dict_params:
        :param n_folds:
        :return:
        '''
        # keep track of evals
        global ITERATION
        ITERATION += 1

        # make sure paramaeters that need integers are integers
        for parameter in list_int_params:
            dict_params[parameter] = int(dict_params[parameter])

        start = timer()
        loss = func_loss(dict_params)
        run_time = timer() - start

        # write to csv file ('a' means append)
        try:
            of_connection = open('{}{}_{}_cv_results.csv'.format(output_dir, target_name, name), 'a', newline='')
            writer = csv.writer(of_connection)
            writer.writerow([loss, dict_params, ITERATION, run_time])
            of_connection.close()

        except PermissionError:
            time.sleep(1)  # if you get a permission error, wait for a second, this will ensure the file is closed
            try:
                of_connection = open('{}{}_{}_cv_results.csv'.format(output_dir, target_name, name), 'a', newline='')
                writer = csv.writer(of_connection)
                writer.writerow([loss, dict_params, ITERATION, run_time])
                of_connection.close()
            except PermissionError:
                print('permission error again')

        return {'loss': loss, 'params': dict_params, 'iteration': ITERATION,
                'train_time': run_time, 'status': STATUS_OK}

    return func_objective


def perform_bayesian_optimisation(max_evals, output_dir, dict_space, alg, list_int_params, train_set, name,
                                  target_name, validation_set=True):
    '''
    Function to optimise the hyperparameter combinations to minimmise a loss function, maximum number of evaluations is
    defined as a trade-off with time to tune
    :param max_evals: maximum number of evaluations of bayesian optimisation
    :param output_dir: directory to save your outputs
    :param dict_space: dictionary defining the hyperparameter space, using hyperopt functions
    :param alg: algorithm you are training, usually input with default hyperparams
    :param list_int_params: hyper parameters that need to be integers
    :param train_set: train data set, including target
    :param name: string name for saving outputs
    :param target_name: name of column containing target
    :param ctrl_cv: Boolean for whether to use cross validation - WIP
    :return:
    '''
    # keep track of results
    bayes_trials = Trials()

    # set up folder for tuning output, within outputs
    tuning_dir = '{}tuning\\'.format(output_dir)

    file_tools.folder_set_up(tuning_dir)

    of_connection = open('{}{}_{}_cv_results.csv'.format(tuning_dir, target_name, name), 'w', newline='')
    writer = csv.writer(of_connection)

    # write headers to file
    writer.writerow(['loss', 'params', 'iteration', 'train_time', 'run_time', 'status'])
    of_connection.close()

    global ITERATION
    ITERATION = 0  # set global iteration variable to zero

    # define objective function using wrapper to set defaults for use in optimisation
    func_objective = func_objective_wrapper(tuning_dir, list_int_params, alg, train_set, target_name, name,
                                            col.parent_id, validation_set=validation_set)

    # run optimisation
    best = fmin(fn=func_objective, space=dict_space, algo=tpe.suggest, max_evals=max_evals, trials=bayes_trials,
                rstate=np.random.RandomState(0))

    bayes_trials_results = sorted(bayes_trials.results, key=lambda x: x['loss'])
    # find ideal hyperparameters
    best_bayes_params = bayes_trials_results[0]

    del best_bayes_params['loss']
    try:
        alg = alg.set_params(**best_bayes_params['params'])
    except catboost.CatBoostError:
        alg = CatBoostClassifier(**best_bayes_params['params'])
    print('{}_{}'.format(name, best_bayes_params))

    return alg, best_bayes_params['params']


def predict_and_extract_shapley_values(trained_model, calibrated_model, X):
    list_features = [*X.columns]

    # predict churn probability
    results = calibrated_model.predict_proba(X)

    # extract shapley values
    model_explainer = shap.TreeExplainer(trained_model)
    df_shap_values = pd.DataFrame(model_explainer.shap_values(X),
                                  columns=[f'shap_{feature}' for feature in list_features])

    # add churn probability as a column
    df_shap_values[col.churn_probability] = results[:, 1]

    # convert expected value from log odds to probability
    expected_val = model_explainer.expected_value
    expected_val_probability = exp(expected_val) / (1 + exp(expected_val))
    df_shap_values['expected_value'] = expected_val_probability

    # convert shap column to probability
    list_shap_columns = [column for column in df_shap_values.columns if 'shap_' in column]

    # calculate the scalar to multiply shapley values by
    df_shap_values['shapley_values_scale'] = (df_shap_values['churn_probability'] - df_shap_values['expected_value']) / \
                                             df_shap_values[
                                                 list_shap_columns].sum(axis=1)

    # scale so that the contributions add to the difference
    for column in list_shap_columns:
        df_shap_values[column] = df_shap_values[column] * df_shap_values['shapley_values_scale']

    df_shap_values.drop(columns='shapley_values_scale', inplace=True)

    return df_shap_values, list_shap_columns
