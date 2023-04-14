"""
__author__ = Radim Musalek
"""

import os
import shutil
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from oranges_case_lab.constants import constants as cnst


def delete_dir_content(*, path_to_dir: str):
    """Delete content of a directory.

    Args:
        path_to_dir (str): Path of the directory whose content is being deleted.
    """

    with os.scandir(path_to_dir) as entries:
        for entry in entries:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry.path)
            else:
                os.remove(entry.path)


def prophet_dafaframe(
        original_dataframe: pd.DataFrame, prophet_ds_column_name: str,
        prophet_y_column_name: str) -> pd.DataFrame:
    """Preprocess original dataframe into Prophet required dataframe structure,
        i.e. datetime ('ds') and value ('y') columns.

    Args:
        original_dataframe (pd.DataFrame): Dataset datetime and value columns to be preprocessed. 
        prophet_ds_column_name (str): Name of the column which will be used as 'ds' column in Prophet.
        prophet_y_column_name (str): Name of the column which will be used as 'y' column in Prophet.

    Returns:
        pandas.DataFrame: Preprocessed dataset in the Prophet required format.
    """

    # selecting only datetime and values columns that will be used in the Prophet dataframe
    prophet_df = original_dataframe[[
        prophet_ds_column_name, prophet_y_column_name]]

    # renaming the columns' names as required by Prophet
    prophet_df.columns = ["ds", "y"]

    # return the preprocessed dataframe
    return prophet_df


def show_interactive_plot(pandas_dataframe: pd.DataFrame, y_column: str, product_type: str) -> px.line:
    """Creates and shows Plotly interactive plot to visulise average prices or quantities development
        of conventional or organic oranges over period of time per region.

    Args:
        pd_dataframe (pd.DataFrame): Dataset including total volumns and avg. prices of
            organic and conventional oranges.
        y_column (str): Expected column names 'AveragePrice' or 'TotalVolume'.
        product_type (str): Expected product type 'conventional' or 'organic'.

    Raises:
        Exception: If y_column is not found in the pandas_dataframe.
        Exception: If y_column isn't 'AveragePrice' or 'TotalVolume'.
        Exception: If product_type isn't 'conventional' or 'organic'.

    Returns:
        px.line: Shows Plotly line chart of product_type's y_column values over time per region.
    """

    # raise an exception if the input y_column name can't be found in the input dataframe
    if y_column not in pandas_dataframe.columns:
        raise Exception("y_column not found in pandas_dataframe")

    # raise an exception if the input y_column name isn't either 'AveragePrice' or 'TotalVolume'
    if y_column not in ['AveragePrice', 'TotalVolume']:
        raise Exception(
            "y_column is expected to be either 'AveragePrice' or 'TotalVolume'")

    # raise an exception if the input product_type isn't either 'conventional' or 'organic'
    if product_type not in ['conventional', 'organic']:
        raise Exception(
            "product_type is expected to be either 'conventional' or 'organic'")

    # plot an interactive chart of the selected product_type 's (filter applied)
    # price or volume (y-axis) over period of time (x-axis)
    fig = px.line(pandas_dataframe[pandas_dataframe['type'] == product_type],
                  x='Date', y=y_column,
                  color='region', title=f"{y_column} of {product_type} per region")
    fig.update_xaxes(dtick='M6', tickformat="%b-%y")

    # return the chart on screen
    return fig.show()


def plot_train_test_pred(
        training_dataframe: pd.DataFrame, test_pred_dataframe: pd.DataFrame,
        pred_parameter: str, product_type: str, model_type: str, save_plot: bool) -> plt:
    """Plots a chart of training, test and predicted values over period of time.

    Args:
        training_dataframe (pd.DataFrame): Dataframe including the training ('y' column) values per date.
        test_pred_dataframe (pd.DataFrame): Dataframe including the test ('y' column) and predicted ('yhat' column) values per date.
        pred_parameter (str): Expected predicted parameter either 'Average Price' or 'Volume'.
        product_type (str): Expected product type either 'Conventional' or 'Organic'.
        model_type (str): Expected model type either 'Default' or 'Fine Tuned'.
        save_plot (bool): True if the plot should be save to outputs dir.

    Raises:
        Exception: If 'y' column is not found in training_dataframe.
        Exception: If 'y' column is not found in test_pred_dataframe.
        Exception: If 'yhat' column is not found in test_pred_dataframe.
        Exception: If the pred_parameter is not either 'Average Price' or 'Volume'.
        Exception: If the product_type is not either 'Conventional' or 'Organic'.
        Exception: If the model_type is not either 'Default' or 'Fine Tuned'.
        Exception: If the save_plot is not a boolean.

    Returns:
        plt: Plots a chart of training, test and predicted values over period of time.
    """

    # raise an exception if the 'y' column name can't be found in the input training_dataframe
    if 'y' not in training_dataframe.columns:
        raise Exception("'y' column not found in training_dataframe")

    # raise an exception if the 'y' column name can't be found in the input test_pred_dataframe
    if 'y' not in test_pred_dataframe.columns:
        raise Exception("'y' column not found in test_pred_dataframe")

    # raise an exception if the 'yhat' column name can't be found in the input test_pred_dataframe
    if 'yhat' not in test_pred_dataframe.columns:
        raise Exception("'yhat' column not found in test_pred_dataframe")

    # raise an exception if the input pred_parameter name isn't either 'Average Price' or 'Volume'
    if pred_parameter not in ['Average Price', 'Volume']:
        raise Exception(
            "pred_parameter is expected to be either 'Average Price' or 'Volume'")

    # raise an exception if the input product_type isn't either 'Conventional' or 'Organic'
    if product_type not in ['Conventional', 'Organic']:
        raise Exception(
            "product_type is expected to be either 'Conventional' or 'Organic'")

    # raise an exception if the input model_type isn't either 'Default' or 'Fine Tuned'
    if model_type not in ['Default', 'Fine Tuned']:
        raise Exception(
            "model_type is expected to be either 'Default' or 'Fine Tuned'")

    # raise an exception if the input save_plot isn't boolean
    if not isinstance(save_plot, bool):
        raise Exception(
            "save_plot is expected to be boolean, i.e. either True or False")

    # create the training plot part
    training_dataframe.set_index('ds')['y'].plot(label='Training')

    # create the test plot part
    test_pred_dataframe.set_index('ds')['y'].plot(label='Test')

    # create the prediction plot part
    test_pred_dataframe.set_index('ds')['yhat'].plot(label='Prediction')

    # define the x-axis label name
    plt.xlabel('Date')

    # define the y-axis label name
    plt.ylabel(pred_parameter)

    # define the title
    plt.title(f"{pred_parameter} {product_type} - {model_type} Model")

    # show legend
    plt.legend()

    # save the plot into the OUTPUTS_DIR if save_plot is True
    if save_plot:
        plt.savefig(str(cnst.OUTPUTS_DIR) +
                    f"/{pred_parameter} {product_type} - {model_type} Model.png")

    # return the plot on screen
    return plt.show()


def plot_forecast(
        original_dataframe: pd.DataFrame, forecast_dataframe: pd.DataFrame,
        pred_parameter: str, product_type: str, save_plot: bool) -> plt:
    """Plots a chart of original and forecasted values over period of time.

    Args:
        original_dataframe (pd.DataFrame): Dataframe including the original ('y' column) values per date.
        forecast_dataframe (pd.DataFrame): Dataframe including the forecasted ('yhat' column) values per date.
        pred_parameter (str): Expected predicted parameter either 'Average Price' or 'Volume'.
        product_type (str): Expected product type either 'Conventional' or 'Organic'.
        save_plot (bool): True if the plot should be save to outputs dir.

    Raises:
        Exception: If 'y' column is not found in original_dataframe.
        Exception: If 'yhat' column is not found in forecast_dataframe.
        Exception: If the pred_parameter is not either 'Average Price' or 'Volume'.
        Exception: If the product_type is not either 'Conventional' or 'Organic'.
        Exception: If the save_plot is not a boolean.

    Returns:
        plt: Plots a chart of original and forecasted values over period of time.
    """

    # raise an exception if the 'y' column name can't be found in the input original_dataframe
    if 'y' not in original_dataframe.columns:
        raise Exception("'y' column not found in original_dataframe")

    # raise an exception if the 'yhat' column name can't be found in the input forecast_dataframe
    if 'yhat' not in forecast_dataframe.columns:
        raise Exception("'yhat' column not found in forecast_dataframe")

    # raise an exception if the input pred_parameter name isn't either 'Average Price' or 'Volume'
    if pred_parameter not in ['Average Price', 'Volume']:
        raise Exception(
            "pred_parameter is expected to be either 'Average Price' or 'Volume'")

    # raise an exception if the input product_type isn't either 'Conventional' or 'Organic'
    if product_type not in ['Conventional', 'Organic']:
        raise Exception(
            "product_type is expected to be either 'Conventional' or 'Organic'")

    # raise an exception if the input save_plot isn't boolean
    if not isinstance(save_plot, bool):
        raise Exception(
            "save_plot is expected to be boolean, i.e. either True or False")

    # create the original values plot part
    original_dataframe.set_index('ds')['y'].plot(label='Original')

    # create the forecasted values plot part
    forecast_dataframe.set_index('ds')['yhat'].plot(label='Forecast')

    # define the x-axis label name
    plt.xlabel('Date')

    # define the y-axis label name
    plt.ylabel(pred_parameter)

    # define the title
    plt.title(f"{pred_parameter} of {product_type} - Forecast")

    # show legend
    plt.legend()

    # save the plot into the OUTPUTS_DIR if save_plot is True
    if save_plot:
        plt.savefig(str(cnst.OUTPUTS_DIR) +
                    f"/{pred_parameter} of {product_type} - Forecast.png")

    # return the plot on screen
    return plt.show()
