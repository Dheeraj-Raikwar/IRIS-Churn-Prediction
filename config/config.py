import os
import pandas as pd
import numpy as np

from hyperopt import hp
from catboost import CatBoostClassifier


class FileLocations:

    def __init__(self):
        self.dir_project = os.getcwd().split('Users\\')[
                               0] + 'Users\\Public\Churn Prediction\\'

        # define input file location and names
        self.dir_input = self.dir_project + '2. Interim Outputs\\'
        self.loc_feature_matrix = self.dir_input + 'feature_matrix.csv'

        self.loc_feature_matrix_train = self.dir_input + 'feature_matrix_prepped_train.pkl'
        self.loc_feature_matrix_out_of_time_test = self.dir_input + 'feature_matrix_prepped_out_of_time_test.pkl'
        self.loc_feature_matrix_correlations_removed = self.dir_input + 'feature_matrix_train_correlations_removed.pkl'
        self.loc_feature_matrix_top_features = self.dir_input + 'feature_matrix_train_top_features.pkl'
        self.loc_feature_matrix_test = self.dir_input + 'feature_matrix_prepped_test.pkl'

        # define output file locations
        self.dir_output = self.dir_project + '4. Python Analysis\\1. Outputs\\'
        self.dir_models = self.dir_project + '4. Python Analysis\\2. Models\\'
        self.dir_checks = self.dir_project + '4. Python Analysis\\3. Checks and Tests\\'

        self.loc_feature_selection = self.dir_checks + 'feature_selection.xlsx'
        self.loc_correlation_test_scores = self.dir_checks + 'correlation_test_scores.xlsx'
        self.loc_highly_correlated_feats = self.dir_checks + 'highly_correlated_feats.csv'
        self.loc_model = self.dir_models + 'model.pkl'
        self.loc_model_calibrated = self.dir_models + 'model_calibrated.pkl'
        self.loc_model_hyperparams = self.dir_models + 'best_hyperparams.pkl'
        self.loc_multiple_model_hyperparams = self.dir_models + 'best_hyperparams_different_models.pkl'
        self.loc_test_train_churn_probabilities = self.dir_checks + 'churn_probabilities_test_train.csv'
        self.loc_model_iterations = self.dir_checks + 'model_feature_iterations.pkl'
        self.loc_dropped_correlated_features_dict = self.dir_checks + 'dropped_correlated_features.pkl'

        #  outputs
        self.loc_predictions_with_shap_values = self.dir_output + 'predictions_with_shap_values.csv'
        self.loc_predictions_with_shap_values_melt = self.dir_output + 'predictions_with_shap_values_melt.csv'
        self.loc_full_dataset_predictions_with_shap_values = self.dir_output + 'full_dataset_shap_values_melt.csv'
        self.loc_shap_averages = self.dir_output + 'shap_averages_for_feature_importance.csv'


class ColumnHeaders:

    def __init__(self, df_feature_matrix=None):

        self.product_family = 'Product_Family'
        self.month = 'Month'
        self.parent_id = 'Parent_Customer_ID'
        self.product_group = 'Product_Group'
        self.child_id = 'Child_ID'
        self.customer_or_product_churn = 'Customer_Or_Product_Churn'
        self.customer_churn = 'Customer_Churn'
        self.product_churn = 'Product_Churn'
        self.churn_probability = 'churn_probability'
        self.list_id_vars = [self.product_family,
                             self.product_group,
                             self.parent_id,
                             self.child_id,
                             self.month]

        self.list_id_vars_not_cat = [self.parent_id,
                                     self.child_id,
                                     self.month,
                                     self.product_group]

        list_not_cat_var_dont_ohe = [self.parent_id,
                                     self.product_group,
                                     self.child_id,
                                     self.month,
                                     'Acquisition Date',
                                     'Month_Minus_1_Year',
                                     'Month_Minus_6_Months',
                                     'Customer_Churn_Month',
                                     'Product_Join_Month',
                                     'First_Date_In_Snowball',
                                     'Month_Minus_2_Years',
                                     'Right_Month_Minus_1_Year',
                                     'Headquarters',
                                     'Region',
                                     'Specialties',
                                     'Industry',
                                     'indus']

        self.dict_label_encode = {'Company_size': {'0-1 employees': 1,
                                                   '2-10 employees': 2,
                                                   '11-50 employees': 3,
                                                   '51-200 employees': 4,
                                                   '201-500 employees': 5,
                                                   '501-1,000 employees': 6,
                                                   '1,001-5,000 employees': 7,
                                                   '5,001-10,000 employees': 8,
                                                   '10,001+ employees': 9},
                                  'NPS_Classification': {'Neutral': 1,
                                                         'Promoter': 2,
                                                         'Detractor': 0}}

        self.list_churn_flags = [self.customer_churn,
                                 self.product_churn,
                                 self.customer_or_product_churn]

        self.list_features_to_exclude = ['Years_Since_First_Date_In_Snowball',
                                         'Months_Since_First_Date_In_Snowball',
                                         'First_Date_In_Snowball_ARR',
                                         'Product_Lifetime']

        if df_feature_matrix is not None:
            self.list_feature_and_target_columns = [column for column in df_feature_matrix.columns if
                                                    column not in self.list_id_vars_not_cat
                                                    and column not in self.list_features_to_exclude]

        self.list_categorical_variables = []
        self.list_categorical_variables_OHE = []
        self.list_categorical_variables_label_encode = []
        self.list_numerical_variables = []

        if df_feature_matrix is not None:
            for column in self.list_feature_and_target_columns:

                # not a target variable of intentionally excluded
                if column not in self.list_churn_flags:
                    datatype = df_feature_matrix[column].dtype
                    count_distinct = len(df_feature_matrix[column].unique())

                    if count_distinct == 2 or datatype in [str, object] and column not in list_not_cat_var_dont_ohe:
                        self.list_categorical_variables.append(column)

                        if count_distinct > 2 and datatype in [str, object] and column not in self.dict_label_encode:
                            self.list_categorical_variables_OHE.append(column)

                            if count_distinct > 5:
                                print(f'Warning, OHE for {column} may create too many columns,'
                                      f' {count_distinct} unique values')

                        else:
                            self.list_categorical_variables_label_encode.append(column)

                    elif datatype in [float, int, 'int64']:
                        self.list_numerical_variables.append(column)

                    elif column not in list_not_cat_var_dont_ohe:
                        print(
                            f'!!! {column} not added to variable list, dtype: {datatype}, count distinct: {count_distinct}')


class ModelConfig:
    def __init__(self):
        col = ColumnHeaders()

        # model settings
        self.target_col = col.customer_or_product_churn
        self.latest_feature_list = [
            'LTM_Upsell_Downsell_Organic_%_ARR', 'No_of_Product_Groups', 'ARR_vs_12mo_ago',
            'Monthly_Delta_Upsell_Organic',
            'L12M_%_Increase_in_Ticket_Count', 'L12M_Average_Days_Ticket_Open', 'L6M_Interactions_Change',
            'Customer_Discount_%_of_ARR', ]
        self.model_name = 'CatBoost'
        from catboost import CatBoostClassifier as model
        self.model = model

        # hyper-params
        self.ctrl_tune_hyperparams = False
        self.class_weights = {0: 1,
                              1: 12}  # weight incorrect classification of churn 12x higher
        self.bayes_evals = 50
        self.n_folds = 1

        # modelling adjustments
        self.split_data_for_out_of_time_test = True
        self.start_train_set_incl = '2019-04-01'
        self.date_for_split = '2020-02-01'  # this is inclusive for the test
        self.max_test_set_incl = '2020-06-01'
        self.calibrate_probabilities = True
        self.test_split = 0.2
        self.rand_state = 0

        # feature selection (baseline model)
        self.iterations_feature_selection = 3
        self.p_val_threshold = 0.01
        self.p_val_thresh_dont_test = 0.4
        self.remove_corr_features_threshold = 0.5
        self.top_features_count = 20  # for baseline model only


    def define_hyperparameters_space(self, alg):
        '''
        Defining the hyperparameter space for each of the ML algorithms used
        :return:
        '''
        dict_hyperparams_space = {}
        dict_hyperparams_space['CatBoost'] = {'params': {'learning_rate': hp.loguniform('learning_rate', np.log(0.01),
                                                                                        np.log(1.0)),
                                                         'depth': hp.quniform('depth', 3, 15, 1),
                                                         'random_state': 0,
                                                         'verbose': False,

                                                         # can't do iterations and estimators
                                                         # 'n_estimators': hp.quniform('n_estimators', 50, 500, 25),
                                                         'iterations': hp.quniform('iterations', 10, 500, 1),
                                                         'l2_leaf_reg': hp.quniform('l2_leaf-reg', 1, 500, 1),

                                                         # number of splits for numerical and cat. features respectively
                                                         'border_count': hp.quniform('border_count', 1, 255, 1),
                                                         # 'ctr_border_count': hp.quniform('ctr_border_count', 1, 10,
                                                         #                                 1),

                                                         # logloss for 2-class classification
                                                         'loss_function': 'Logloss',

                                                         # built in facility to prevent overfitting
                                                         'use_best_model': True,
                                                         'eval_metric': 'AUC'},

                                              'list_integer_params': ['depth',
                                                                      'l2_leaf_reg',
                                                                      'border_count',
                                                                      # 'ctr_border_count',
                                                                      'iterations']}

        dict_hyperparams_space['XGBoost'] = {'params': {'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                                                        'gamma': hp.uniform('gamma', 0, 9),
                                                        'learning_rate': hp.loguniform('learning_rate', np.log(0.01),
                                                                                       np.log(1.0)),
                                                        'max_depth': hp.quniform('max_depth', 3, 15, 1),
                                                        'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
                                                        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                                                        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
                                                        'subsample': hp.uniform('subsample', 0.7, 1.0),
                                                        'random_state': 0,
                                                        'n_estimators': hp.quniform('n_estimators', 50, 500, 1),
                                                        'eval_metric': 'auc'},
                                             'list_integer_params': ['max_depth', 'min_child_weight', 'n_estimators']}

        dict_hyperparams_space['LightGBM'] = {'params': {'objective': 'binary',
                                                         'learning_rate': hp.loguniform('learning_rate', np.log(0.01),
                                                                                        np.log(1.0)),
                                                         'early_stopping_round': hp.quniform('early_stopping_round', 10,
                                                                                             100, 1),
                                                         'max_depth': hp.quniform('depth', 3, 15, 1),
                                                         'lambda_l1': hp.uniform('lambda_l1', 0.0, 1.0),
                                                         'feature_fraction': hp.uniform('feature_fraction', 0.0, 1.0),
                                                         'random_state': 0,
                                                         'verbose': 0,

                                                         'num_iterations': hp.quniform('iterations', 10, 500, 1),

                                                         'metric': 'AUC',
                                                         'sample_pos_weight': 10},

                                              'list_integer_params': ['max_depth',
                                                                      'early_stopping_round',
                                                                      'num_iterations']}

        return dict_hyperparams_space[alg]['params'], dict_hyperparams_space[alg]['list_integer_params']


class Requirements:
    """
    Define the required settings for the project, including required packages for the projec to run
    """

    def __init__(self):
        self.required_packages = ['pandas']
