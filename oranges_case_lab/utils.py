"""
__author__ = Radim Musalek
"""

import os
import shutil
import pandas


def delete_dir_content(*, path_to_dir: str):
    '''
    Delete content of a directory
    '''
    with os.scandir(path_to_dir) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)


def prophet_dafaframe(original_dataframe: pandas.DataFrame, prophet_ds_column_name: str, prophet_y_column_name: str) -> pandas.DataFrame:
    '''
    Prepare original dataframe into Prophet required dataframe structure,
    i.e. datetime ("ds") and value ("y") columns
    '''
    prophet_df = original_dataframe[[
        prophet_ds_column_name, prophet_y_column_name]]
    prophet_df.columns = ["ds", "y"]
    return prophet_df
