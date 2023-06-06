# ETL_modules to be used to connect to Azure DB
# Created date 23/09/19

import os
import numpy as np
import time

import urllib
import pandas as pd
import sqlalchemy
import pyodbc

# -- Functions --------------------------------------------------------------------------------------------------- #


def db_connection():
    server = 'db-name.database.windows.net'
    database = 'DB-NAME'
    username = 'User'
    password = 'password'
    driver = '{SQL Server Native Client 11.0}'
    cnxn = pyodbc.connect(
        'DRIVER=' + driver + ';PORT=1433;SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username
        + ';PWD=' + password)

    return cnxn


def load_sql_data(Query):
    """ Loads a SQL query using the credentials specified in db_connection(). """
    t0 = time.time()
    print('Querying SQL Server...', end='')
    # Read from db and display time to extract data into pandas dataframe
    data = pd.read_sql(Query, con=db_connection())
    t1 = time.time()
    print(' complete! {:.2f}s'.format(t1 - t0))
    return data


def load_table(table_name):
    """
    Loads a table "Tablename" from specified DB ("BG") on specified SQL SERVER ("tdr-db.database.windows.net")
    Table is loaded and returned as Pandas Dataframe
    """

    t0 = time.time()

    # Read from db and display time to extract data into pandas dataframe
    data = pd.read_sql('select * from ' + table_name, con=db_connection())
    t1 = time.time()
    print('Querying SQL Server: ', np.round(t1 - t0), ' seconds')
    return data


def load_and_save_table(table_name, pickle_loc, refresh=True):
    if refresh is True:
        # Read from DB
        print('Loading ' + table_name + ' from SQL')
        data = load_table(table_name)
        # Write to pickle
        print('Writing ' + table_name + ' to pickle')
        data.to_pickle(pickle_loc + table_name + '.pkl')

    else:
        try:
            print('Reading ' + table_name + ' from pickle')
            data = pd.read_pickle(pickle_loc + table_name + '.pkl')
        except FileNotFoundError:
            print('Cannot locate ' + table_name + ' pickle')
            print('Loading ' + table_name + ' from SQL')
            data = load_table(table_name)
            print('Writing ' + table_name + ' to pickle')
            data.to_pickle(pickle_loc + table_name + '.pkl')

    return data


def load_and_save_query(query, pickle_loc, file_name, refresh=True):
    if refresh is True:
        # Read from DB
        print('Loading ' + file_name + ' from SQL')
        data = load_sql_data(query)
        # Write to pickle
        print('Writing ' + file_name + ' to pickle')
        data.to_pickle(pickle_loc + file_name + '.pkl')

    else:
        try:
            print('Reading ' + file_name + ' from pickle')
            data = pd.read_pickle(pickle_loc + file_name + '.pkl')
        except FileNotFoundError:
            print('Cannot locate ' + file_name + ' pickle')
            print('Loading ' + file_name + ' from SQL')
            data = load_sql_data(query)
            print('Writing ' + file_name + ' to pickle')
            data.to_pickle(pickle_loc + file_name + '.pkl')

    return data


def get_SQL_table_if_does_not_exist(save_loc, table_name):
    """ Pulls specific table from SQL database and stores to specific .pkl path  """

    # for reference date check if file loc exists
    file_loc = save_loc + '{}.pkl'.format(table_name)

    if os.path.exists(file_loc) is True:
        df_sql = pd.read_pickle(save_loc + '{}.pkl'.format(table_name))

    else:
        print('Table {} not saved locally. Querying DB now.'.format(table_name))
        df_sql = load_sql_data('SELECT * FROM [dbo].[{}]'.format(table_name))
        df_sql.to_pickle(save_loc + '{}.pkl'.format(table_name))  # save for future use

    return df_sql


def download_table():
    output_loc = project_loc + 'forecasting\\outputs\\'

    n_rows = 1_000_000_000

    qry = "SELECT TOP {} * FROM [dbo].[JMAN_AH_sales_by_article]".format(n_rows)

    df = load_sql_data(qry)

    df.to_pickle(output_loc + "190923_JMAN_AH_sales_by_article.pkl")


def list_tables(db_con):
    cursor = db_con.cursor()

    tables_ref = pd.DataFrame().from_records(cursor.tables())
    tables_ref.columns = ['db_name', 'db_ref', 'item_name', 'item_type', '-']

    return tables_ref


def write_to_excel(df_dict, save_loc_file_name):
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(save_loc_file_name, engine='xlsxwriter')

    for df_name in df_dict.keys():
        df_dict[df_name].to_excel(writer, sheet_name=df_name[:30])

    # Close the Pandas Excel writer and output the Excel file.
    writer.save()


def create_table_summaries(output_loc, refresh=True):
    if refresh is True:

        nrows = 1000

        db_cnxn = db_connection()

        table_ls = list_tables(db_cnxn)

        tables_to_query = table_ls[['db_ref', 'item_name']].loc[table_ls['db_ref'].isin(['dbo', 'his'])]

        tables_data = {}
        for tb_name, db_ref in zip(tables_to_query['item_name'], tables_to_query['db_ref']):
            qry = "SELECT TOP {} * FROM [{}].[{}]".format(nrows, db_ref, tb_name)
            tables_data[tb_name] = load_sql_data(qry)

        write_to_excel(tables_data, output_loc + 'sql_db_summary.xlsx')


def save_DF_to_DB(df, tablename='Prediction Results'):
    '''
    Loads DF into TDR SQL SERVER DB by replacing existing table
    SQL alchemy can throw an error with datetime objects

    '''

    t0 = time.time()
    print('writing to DB')
    server = 'dbname.database.windows.net'
    database = 'DBNAME'
    username = 'user'
    password = 'password'

    driver = '{SQL Server Native Client 11.0}'

    params = urllib.parse.quote_plus(
        'DRIVER=' + driver + ';PORT=1433;SERVER=' + server + ';PORT=1443;DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)

    if "index" in df.columns.values:
        df = df.drop(columns=["index"])

    tsql_chunksize = 2097 // len(df.columns)

    df.to_sql(tablename, engine, if_exists='replace', index=False, chunksize=tsql_chunksize)
    t1 = time.time()
    print('Writing results To SQL Server: ', t1 - t0, ' seconds')


def parse_sql_query(q, placeholder_to_replace='', reference=''):
    """ Parse SQL query from nice-to-read format to easy-to-query format. Removes new line characters.
    NOTE: q should not contain any SQL comments"""
    return q.replace('\n', ' ').replace(placeholder_to_replace, str(reference))


if __name__ == '__main__':
    # download_table()

    create_table_summaries()
