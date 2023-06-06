import os
import pandas as pd


class FileLocationTools:

    def folder_set_up(self, folder_path):
        """
        if folder path does not exist, create it
        :param folder_path: Location of new folder in format "...location\\new_folder"
        :return: String of file path to new folder
        """

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)


    def find_latest_file(self, file_string, file_path):
        """
        find latest file with rile name
        :param file_string:
        :param file_path:
        :return:
        """
        # TODO: update this function so that it can handle versions over 10
        # Note that this function won't work where there are files in the location which have this filename as a
        # substring
        list_of_files = os.listdir(file_path)  # get list of files in location

        # allow input of path sting
        if file_path in file_string:
            file_string = file_string.split(file_path)[1]

        # allow input of file name with file type e.g. ".csv"
        if '.' in file_string:
            file_string = file_string.split('.')[0]

        try:
            # spaces in the condition is for the case where you want an unversioned file as the latest version also
            # this will likely be the case for e.g. an output going into Tableau, where the name of the source matters
            matching_files = [x for x in list_of_files if (f' {file_string} v' in x) or (f'{file_string}_v' in x)]
            matching_files.sort()
            latest_file_name = matching_files[-1]
            version = int(latest_file_name.replace(file_string, '').split('.')[0].split('v')[-1])
            file_found = True
        except IndexError:
            file_found = False
            latest_file_name = None
            version = None

        return file_found, version, latest_file_name

    def define_output_file_name(self, file_string, file_path):
        '''
        Define output file name with version control -> date and version
        :param file_string: name of file to be saved (with or without file path)
        there should be no versions or dates in this input
        :param file_path: location this file will be saved
        :return: output file name in the format 'date file_string vX' where X is the latest version of that file name
        '''

        def get_date_for_file_outputs():
            """
            :return: String of today's date e.g. '200219'
            Useful for creating output folders for each day's work
            See function below
            """
            t0 = pd.Timestamp.now()
            date_str = t0.strftime('%y%m%d')

            return date_str

        # allow input of path sting
        if file_path in file_string:
            file_string = file_string.split(file_path)[1]

        # allow input of file name with file type e.g. ".csv"
        if '.' in file_string:
            file_string = file_string.split('.')[0]

        file_found, version, _ = self.find_latest_file(file_string, file_path)

        # if file has been found, update version, else +1 to version
        if file_found:
            new_file_name = file_string + ' v' + str(version + 1)
        else:
            new_file_name = file_string + ' v1'

        # update date
        new_file_name = get_date_for_file_outputs() + ' ' + new_file_name
        print('output saved as {}'.format(new_file_name))

        return new_file_name


def data_checks(data, df_name_as_string, output_loc):
    '''
    Runs a series of checks on your dataframe and exports information on the quality of the columns in your dataframe
    :param data: Dataframe to be checked
    :param df_name_as_string: Name of dataframe for exporting csv of data checks
    :param output_loc: Location the data checks file should be saved
    '''

    import numpy as np

    number_of_rows = data.size
    print("Check total number of rows in {}".format(df_name_as_string) + str(number_of_rows))

    if number_of_rows > 10000:
        print("Dataset is large (greater than 10,000 rows), saving csv file of the first 100 rows.")
        data_100_rows = data.head(100)
        data_100_rows.to_csv(output_loc + '{}_first_100_rows.csv'.format(df_name_as_string))

    print("Check what columns we have in {}".format(df_name_as_string))
    print(data.columns)

    new_df = pd.DataFrame()
    new_df["column"] = data.columns
    new_df = new_df.set_index('column')
    new_df["column_type"] = data.dtypes
    new_df["na_count"] = data.isna().sum(numeric_only=True)
    new_df["na_percentage"] = data.isna().sum(numeric_only=True) / number_of_rows
    new_df["number_of_unique_observations"] = data.nunique()
    new_df["min_value"] = data.min(numeric_only=True)
    new_df["max_value"] = data.max(numeric_only=True)
    new_df["sum"] = data.sum(numeric_only=True)
    new_df["na_percentage"] = np.where(new_df["number_of_unique_observations"] == 0, 1, new_df["na_percentage"])
    # analyses of date columns to exclude dates = 1900.01.01
    new_df["perc_non_zero_dates"] = np.nan
    new_df["min_non_zero_dates"] = np.nan
    new_df["max_non_zero_dates"] = np.nan

    for col in data.columns:
        if data[col].dtype == 'datetime64[ns]':
            data_sample = data[data[col] > pd.to_datetime('1900.01.01')]
            non_zero_dates = data_sample.size
            new_df.loc[col, ["perc_non_zero_dates"]] = non_zero_dates / number_of_rows
            date_min = data_sample[col].min()
            new_df.loc[col, ["min_non_zero_dates"]] = date_min
            date_max = data_sample[col].max()
            new_df.loc[col, ["max_non_zero_dates"]] = date_max

    print("Saving csv file of data checks")
    new_df.to_csv(output_loc + "{}_data_checks.csv".format(df_name_as_string))


def write_dfs_to_excel(dict_of_dfs_to_export, output_loc):
    '''
    Function for quickly exporting multiple dataframes to sheets of an excel
    :param dict_of_dfs_to_export: A dictionary which contains the sheet name as the key and the dataframe as the value
    :param output_loc: The location where you want the output excel to be saved, should include the file extension, eg. '.xlsx'
    :return:
    '''

    xlsx_writer = pd.ExcelWriter(output_loc)

    for sheetname, df in dict_of_dfs_to_export.items():
        df.to_excel(xlsx_writer, sheet_name=sheetname, float_format='%0.4f')
        worksheet = xlsx_writer.sheets[sheetname]
        for idx, column in enumerate(df.columns):
            series = df[column]
        max_len = series.astype(str).map(len).max() + 1  # len of the largest item and extra space
        worksheet.set_column(idx + 1, idx + 1, max_len)  # set column width
        max_index = df.index.astype(str).map(len).max()
        worksheet.set_column(0, 0, max_index)  # set the width of the index column

    xlsx_writer.save()


if __name__ == "__main__":

    # --- 1. FileLocations tools examples ---------------------------------------------------------------------------- #
    """
    Examples of using file location tools
    For all of these examples, you will first need to import and instantiate the FileLocations class from config
    """

    from config.config import FileLocations
    from tools.tools import FileLocationTools

    file_locations = FileLocations()
    file_tools = FileLocationTools()

    # --- 1.a. set up folders example -------------------------------------------------------------------------------- #
    """
    This script will set up your folder setup for you
    Script will then setup the folder structure (feel free to adjust the below to whatever floats your boat
    """

    # define list of folders to set up in project directory
    list_folders = ['0. Admin', '1. Raw Data', '2. Analysis', '3. Meeting Materials', '4. Deliverables']

    # loop through creating folders if they do not exist already
    for folder in list_folders:
        file_tools.folder_set_up(file_locations.dir_project + folder)

    # --- 1.b. saving dated and versioned file example --------------------------------------------------------------- #
    """
    For files where it would be useful for you to keep a record of previous outputs, and know the version and date 
    of the latest files e.g. final outputs, it will be useful to automate the dating and version control of your outputs
    NOTE: take care with using this on large files as it can very quickly clog up Sharepoint
    """

    # define file name without date and version
    file_name = 'final_outputs_versioned_example'

    # set up folder for saving outputs
    dir_example_outputs = file_locations.dir_output + 'examples\\'

    file_tools.folder_set_up(dir_example_outputs)

    # create dummy data frame to save outputs
    df_dummy_data = pd.DataFrame(data=[[0, 1, 2],
                                       [3, 4, 5]],
                                 columns=['col1', 'col2', 'col3'])

    # find latest version of file and add 1 to create new version with today's date
    file_name_versioned = file_tools.define_output_file_name(dir_example_outputs + file_name)

    # save output to csv - note the define output file name is file type agnostic, so you need to add the .csv here
    df_dummy_data.to_csv(dir_example_outputs + file_name_versioned + '.csv')

    # --- 2. Export to excel example --------------------------------------------------------------------------------- #
    '''
    Use of a function to export to excel
    '''
    # create dummy dataframes for the example to export to an excel
    df_a = pd.DataFrame.from_dict(
        {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd'], 'col_3': ['apple', 'banana', 'orange', 'grape']})
    df_b = pd.DataFrame.from_dict({'col_A': [1, 1, 1, 0], 'col_B': [0, 1, 1, 0], 'col_C': [0, 1, 0, 1]})
    # create a dictionary of these dataframes and sheet names for export
    # NOTE: The sheetname must be listed first in the dictionary followed by the df
    dict = {'Dataframe A': df_a,
            'Dataframe B': df_b}

    # Name the file I want to export and use the define output file name to get the correct version
    filename = file_tools.define_output_file_name(dir_example_outputs + 'Exporting_to_Excel_Example')

    # Write the dataframes to different sheets of the excel and save
    write_dfs_to_excel(dict, dir_example_outputs + filename + '.xlsx')
