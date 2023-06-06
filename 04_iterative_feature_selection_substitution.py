'''
Author: Freddie Hampel, JMAN Group
Date: April 2021

Runs after 01_feature_matrix_prep.py

Script to:
1. iterate through features with add one, remove one to see if accuracy improves
2. swap in and out correlated features to keep the best one

Note: suggested improvement would be to update 1. to include consideration feature importance
'''

from catboost import CatBoostClassifier
import csv

import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.metrics import roc_auc_score

from tools.modelling_functions import split_data_into_test_train_wrapper

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

# --- instantiate classes -------------------------------------------------------------------------------------------- #

file_loc = FileLocations()
df_feature_matrix_train_with_correlations_removed = pd.read_pickle(file_loc.loc_feature_matrix_correlations_removed)
df_feature_matrix_full_train_set = pd.read_pickle(file_loc.loc_feature_matrix_train)
df_feature_matrix_train_top_features = pd.read_pickle(file_loc.loc_feature_matrix_top_features)

col = ColumnHeaders(df_feature_matrix_train_with_correlations_removed)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()


# --- functions ----------------------------------------------------------------------------------------------------- #


def iterative_feature_selection(df_feature_matrix,
                                list_current_features,
                                threshold_for_improvement,
                                iterations,
                                output_dir=file_loc.dir_output,
                                file_name='feature_step_iteration_results',
                                roc_auc_best=0,
                                improve_method_add=True,
                                counter_trained=0):

    list_all_features = [column for column in df_feature_matrix.columns if
                         column not in [model_config.target_col] + col.list_id_vars]

    counter = 0
    feature_new = None
    feature_remove = None
    list_missing_features = [x for x in list_all_features if x not in list_current_features]
    n = 0
    list_feature_list_best = list_current_features

    df_test, df_train = split_data_into_test_train_wrapper(df_feature_matrix)
    df_val, df_train = split_data_into_test_train_wrapper(df_train)

    list_all_features = [column for column in df_train if column not in col.list_id_vars + [model_config.target_col]]

    X_train = df_train[list_all_features]
    X_test = df_test[list_all_features]
    X_val = df_val[list_all_features]

    y_train = df_train[model_config.target_col]
    y_test = df_test[model_config.target_col]
    y_val = df_val[model_config.target_col]

    while (counter_trained <= iterations) & (n < 1000):  # add n conditions to stop infinite loop
        if (improve_method_add & (counter < len(list_missing_features))) | (
                (not improve_method_add) & (counter < len(list_current_features))):
            ctrl_train = True

            if n > 0:

                if improve_method_add:
                    feature_new = list_missing_features[counter]

                    try:
                        while feature_new in list_current_features:
                            counter += 1
                            feature_new = list_missing_features[counter]
                        list_current_features.append(feature_new)

                    except IndexError:
                        ctrl_train = False
                        improve_method_add = not improve_method_add
                        list_missing_features = [x for x in list_all_features if x not in list_current_features]

                else:
                    feature_remove = list_current_features[counter]
                    list_current_features.remove(feature_remove)

        else:
            ctrl_train = False
            improve_method_add = not improve_method_add
            list_missing_features = [x for x in list_all_features if x not in list_current_features]
            counter = 0  # reset counter

        if ctrl_train:
            model = CatBoostClassifier(iterations=500,
                                       verbose=False,
                                       eval_metric='AUC',
                                       class_weights=model_config.class_weights,
                                       # logloss for 2-class classification
                                       loss_function='Logloss',
                                       # built in facility to prevent overfitting
                                       use_best_model=True)

            model.fit(X_train[list_current_features], y_train, eval_set=(X_val[list_current_features], y_val))

            roc_auc_test = roc_auc_score(y_test, model.predict(X_test[list_current_features]))
            roc_auc_train = roc_auc_score(y_train, model.predict(X_train[list_current_features]))

            if improve_method_add:

                add_remove = 'Add'
                feature_for_output = feature_new
                if n > 0:
                    counter += 1

                if roc_auc_test > roc_auc_best + threshold_for_improvement:
                    status_improvement = 'Y'
                    roc_auc_best = roc_auc_test
                    list_feature_list_best = list_current_features

                elif n > 0:
                    list_current_features.remove(feature_new)
                    status_improvement = 'N'

                else:
                    status_improvement = 'N'

            else:
                add_remove = 'Remove'
                feature_for_output = feature_remove

                if roc_auc_test > roc_auc_best - threshold_for_improvement:  # preference for fewer features
                    status_improvement = 'Y'
                    roc_auc_best = roc_auc_test
                    list_feature_list_best = list_current_features

                elif n > 0:
                    list_current_features.insert(counter, feature_remove)
                    counter += 1
                    status_improvement = 'N'

                else:
                    status_improvement = 'N'

            # write to csv file ('a' means append)
            csv_connection = open(f'{output_dir}{file_name}_v{version}.csv', 'a', newline='')
            writer = csv.writer(csv_connection)
            writer.writerow([n, roc_auc_test, roc_auc_train, list_current_features, feature_for_output, add_remove,
                             status_improvement])

            counter_trained += 1
        n += 1
        print(n, counter_trained)

    return list_feature_list_best, roc_auc_best, improve_method_add, counter_trained


def test_substitution_of_correlated_features(df_feature_matrix,
                                             output_dir,
                                             list_best_features,
                                             dict_dropped_features,
                                             file_name='correlated_feature_swap_results',
                                             threshold_for_improvement=0.01,
                                             roc_auc_best=0):
    _, version, _ = file_loc_tools.find_latest_file(f'{output_dir}{file_name}', output_dir)
    if version is not None:
        version += 1
    else:
        version = 1

    csv_connection = open(f'{output_dir}{file_name}_v{version}.csv', 'w', newline='')
    writer = csv.writer(csv_connection)
    # write headers to file
    writer.writerow(
        ['n', 'Test roc-auc', 'Train roc-auc', 'Features', 'Removed', 'Added', 'Improvement'])
    csv_connection.close()

    df_test, df_train = split_data_into_test_train_wrapper(df_feature_matrix)
    df_val, df_train = split_data_into_test_train_wrapper(df_train)

    X_train = df_train[[column for column in df_train if column not in [col.parent_id, model_config.target_col]]]
    X_test = df_test[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]
    X_val = df_val[[column for column in df_test if column not in [col.parent_id, model_config.target_col]]]

    y_train = df_train[model_config.target_col]
    y_test = df_test[model_config.target_col]
    y_val = df_val[model_config.target_col]

    n = 1
    if roc_auc_best == 0:
        model = CatBoostClassifier(iterations=500,
                                   verbose=False,
                                   eval_metric='AUC',
                                   class_weights=model_config.class_weights,
                                   # logloss for 2-class classification
                                   loss_function='Logloss',
                                   # built in facility to prevent overfitting
                                   use_best_model=True)

        model.fit(X_train[list_best_features], y_train, eval_set=(X_val[list_best_features], y_val))

        roc_auc_test = roc_auc_score(y_test, model.predict(X_test[list_best_features]))
        roc_auc_train = roc_auc_score(y_train, model.predict(X_train[list_best_features]))
        roc_auc_best = roc_auc_test

        csv_connection = open(f'{output_dir}{file_name}_v{version}.csv', 'a', newline='')
        writer = csv.writer(csv_connection)
        writer.writerow([0, roc_auc_test, roc_auc_train, list_current_features, '-', '-',
                         '-'])

    # loop through testing correlated features
    for feature in tqdm(dict_dropped_features):
        print(f'feature {feature}')

        for replacement_feature in tqdm(dict_dropped_features[feature]):
            print(f'potential replacement {replacement_feature}')

            list_test_features = list_best_features.copy()

            if feature in list_best_features:  # remove feature it is replacing, if it's in the list
                list_test_features.remove(feature)

            else:
                feature = None

            # if original feature not in the list, it's still worth trying the replacements
            list_test_features.append(replacement_feature)

            model = CatBoostClassifier(iterations=500,
                                       verbose=False,
                                       eval_metric='AUC',
                                       class_weights=model_config.class_weights,
                                       # logloss for 2-class classification
                                       loss_function='Logloss',
                                       # built in facility to prevent overfitting
                                       use_best_model=True)

            model.fit(X_train[list_test_features], y_train, eval_set=(X_val[list_test_features], y_val))

            roc_auc_test = roc_auc_score(y_test, model.predict(X_test[list_test_features]))
            roc_auc_train = roc_auc_score(y_train, model.predict(X_train[list_test_features]))

            print(f'roc_auc_test: {roc_auc_test}')

            if roc_auc_test > roc_auc_best + threshold_for_improvement:
                status_improvement = 'Y'
                print(
                    f'{feature} replaced by {replacement_feature} due to an increase in roc_auc of'
                    f' {roc_auc_test - roc_auc_best}')

                roc_auc_best = roc_auc_test
                list_best_features = list_test_features
                feature = replacement_feature

                # note this assumes that all features in this loop are correlated with each other because they are
                # all correlated with the original feature they are replacing

            else:
                status_improvement = 'N'

            # write to csv file ('a' means append)
            csv_connection = open(f'{output_dir}{file_name}_v{version}.csv', 'a', newline='')
            writer = csv.writer(csv_connection)
            writer.writerow([n, roc_auc_test, roc_auc_train, list_current_features, feature, replacement_feature,
                             status_improvement])
            n += 1

    return list_best_features, roc_auc_best


# --- script --------------------------------------------------------------------------------------------------------- #

df_test, df_train = split_data_into_test_train_wrapper(df_feature_matrix_train_with_correlations_removed)
df_val, df_train = split_data_into_test_train_wrapper(df_train)

list_full_features_correlations_removed = [column for column in df_train if
                                           column not in col.list_id_vars + [model_config.target_col]]
list_current_features = [column for column in df_feature_matrix_train_top_features if
                         column not in col.list_id_vars + [model_config.target_col]]


X_train = df_train[list_full_features_correlations_removed]
X_test = df_test[list_full_features_correlations_removed]
X_val = df_val[list_full_features_correlations_removed]

y_train = df_train[model_config.target_col]
y_test = df_test[model_config.target_col]
y_val = df_val[model_config.target_col]

model = CatBoostClassifier(iterations=500,
                           verbose=False,
                           eval_metric='AUC',
                           class_weights=model_config.class_weights,
                           # logloss for 2-class classification
                           loss_function='Logloss',
                           # built in facility to prevent overfitting
                           use_best_model=True)

model.fit(X_train[list_current_features], y_train, eval_set=(X_val[list_current_features], y_val))
roc_auc_best = roc_auc_score(y_test, model.predict(X_test[list_current_features]))

counter_trained = 0
improve_method_add = True
file_name = 'feature_step_iteration_results'

_, version, _ = file_loc_tools.find_latest_file(f'{file_loc.dir_output}{file_name}', file_loc.dir_output)
if version is not None:
    version += 1
else:
    version = 1

csv_connection = open(f'{file_loc.dir_output}{file_name}_v{version}.csv', 'w', newline='')
writer = csv.writer(csv_connection)
# write headers to file
writer.writerow(
    ['n', 'Test roc-auc', 'Train roc-auc', 'Features', 'Feature', 'Add/Remove', 'Improvement'])
csv_connection.close()

while counter_trained <= 500:
    list_current_features, roc_auc_best, improve_method_add, counter_trained = iterative_feature_selection(
        df_feature_matrix_train_with_correlations_removed,
        list_current_features,
        threshold_for_improvement=0.005,
        iterations=500,
        roc_auc_best=roc_auc_best,
        improve_method_add=improve_method_add,
        counter_trained=counter_trained)

    improve_method_add = not improve_method_add

print(f'First round iterations complete, roc_auc best: {roc_auc_best}')

# import correlated features dictionary
with open(file_loc.loc_dropped_correlated_features_dict, 'rb') as file:
    dict_dropped_features = pickle.load(file)
    file.close()

list_best_features_2, roc_auc_best_2 = test_substitution_of_correlated_features(df_feature_matrix_full_train_set,
                                                                                file_loc.dir_output,
                                                                                list_current_features,
                                                                                dict_dropped_features,
                                                                                roc_auc_best=roc_auc_best,
                                                                                threshold_for_improvement=0.005)

print(f'Second round iterations complete, roc_auc best: {roc_auc_best_2}')

print('finished')
