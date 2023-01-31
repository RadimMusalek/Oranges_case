"""
__author__ = Radim Musalek
"""

import os
import shutil
import pandas
import plotly.express as px


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
    Preprocess original dataframe into Prophet required dataframe structure,
    i.e. datetime ("ds") and value ("y") columns
    '''

    prophet_df = original_dataframe[[
        prophet_ds_column_name, prophet_y_column_name]]
    prophet_df.columns = ["ds", "y"]
    return prophet_df


def show_interactive_plot(pandas_dataframe: pandas.DataFrame, y_column: str, product_type: str) -> px.line:
    '''
    Creates and shows Plotly interactive plot to visulise average prices or quantities development
    of conventional or organic oranges over period of time per region.

    Parameters
    ----------
    y_column : str
        Expected column names "AveragePrice" or "TotalVolume"
    product_type : str
        Expected product type "conventional" or "organic"

    Raises
    ------
    Exception
        If y_column is not found in the pandas_dataframe
    Exception
        If y_column isn't 'AveragePrice' or 'TotalVolume'
    Exception
        If product_type isn't 'conventional' or 'organic'

    Returns
    -------
    Shows Plotly line chart of product_type's y_column values over time per region
    '''

    if y_column not in pandas_dataframe.columns:
        raise Exception("y_column not found in pandas_dataframe")

    if y_column not in ["AveragePrice", "TotalVolume"]:
        raise Exception(
            "y_column is expected to be either 'AveragePrice' or 'TotalVolume'")

    if product_type not in ["conventional", "organic"]:
        raise Exception(
            "product_type is expected to be either 'conventional' or 'organic'")

    fig = px.line(pandas_dataframe[pandas_dataframe["type"] == product_type],
                  x="Date", y=y_column,
                  color="region", title=f"{y_column} of {product_type} per region")
    fig.update_xaxes(dtick="M6", tickformat="%b-%y")

    return fig.show()
