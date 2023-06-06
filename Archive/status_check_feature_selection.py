import pickle
import pandas as pd

from config.config import FileLocations, ColumnHeaders, ModelConfig
from tools.tools import FileLocationTools

file_loc = FileLocations()
df_feature_matrix = pd.read_pickle(file_loc.loc_feature_matrix_correlations_removed)

col = ColumnHeaders(df_feature_matrix)
model_config = ModelConfig()
file_loc_tools = FileLocationTools()

with open(file_loc.loc_model_iterations, 'rb') as file:
    dict_results = pickle.load(file)
    file.close()

print('finished')
