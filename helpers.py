import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup


def convert_year(input_year_list):
    """_summary_

    Args:
        input_year_list (Series): _description_

    Returns:
        list: _description_
    """
    added_list = []
    for i in range(len(input_year_list)):
        month = float(input_year_list[i][1]) / 12
        year = input_year_list[i][0]
        added_list.append(float(year) + month)
    return added_list


def get_dataframe(wikiurl):
    """_summary_

    Args:
        wikiurl (str): _description_

    Returns:
        DataFrame: _description_
    """

    # get the response in the form of html
    table_class = "mw-tabular sortable jquery-tablesorter"
    response = requests.get(wikiurl)
    # print(response.status_code)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    weather_table = soup.find('table', {'class': "mw-tabular"})

    df = pd.read_html(str(weather_table))
    # convert list to dataframe
    df = pd.DataFrame(df[0])

    return df


def plot_single_series(ax, x_data, y_data, y_label, color, scatter_type):
    """_summary_

    Args:
        ax (Axes): _description_
        x_data (str): _description_
        y_data (str): _description_
        y_label (str): _description_
        color (str): _description_
        scatter_type (str): _description_

    Returns:
        Axes: _description_
    """
    ax.set_ylabel(y_label, color=color)
    ax.plot(x_data, y_data, scatter_type, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    return ax


def filter_month(data, month_num):
    """_summary_

    Args:
        data (DataFrame): _description_
        month_num (int): _description_

    Returns:
        DataFrame: _description_
    """
    series_length = int(data.shape[0])

    for drop_index in range(1, 13):
        if drop_index == month_num:
            continue
        range_to_drop = range(drop_index - 1, series_length, 12)
        data.drop(range_to_drop, axis=0, inplace=True)

    return data


def find_best_fit(ax, x_dataset, y_dataset, color, line_type='-'):
    """_summary_

    Args:
        ax (Axes): Axes object to plot the line of best fit on.
        x_dataset (Series): Series of 1 set of x axis data.
        y_dataset (Series): Series of 1 set of y axis data.
        color (str): Color of best fit line.
        line_type (str, optional): Style of line for the line of best fit.
            Defaults to '-'.
    """
    z = np.polyfit(x_dataset.values.flatten(), y_dataset.values.flatten(), 1)
    p = np.poly1d(z)

    ax.plot(x_dataset, p(x_dataset), line_type, color=color)


def plot_double_scatter(data, x_data, y_data1, y_data2='', x_label='Year',
                        y_label1='Missing Label', y_label2='Missing Label',
                        color1='red', color2='blue', scatter_type='',
                        fit='none', month_num=0):
    """Takes a DataFrame of weather data and 1-2 column names, along with some
    other optional arguments, and plots the given columns. Can also create lines
    of best fit for the plots.

    Args:
        data (DataFrame): DataFrame of weather data.
        x_data (str): DataFrame column label for x dataset. Expects 'date' or
            or 'avgLowTemp' format.
        y_data1 (str): DataFrame column label for 1st y dataset. Expects 'date'
            or 'avgLowTemp' format.
        y_data2 (str, optional): DataFrame column label for 2nd y dataset.
            Expects 'date' or 'avgLowTemp' format. Defaults to '', wherein the
            function only plots 1 dataset.
        x_label (str, optional): Label for x axis. Defaults to 'Year'.
        y_label1 (str, optional): Label for 1st y axis. Ensure that units are
            included in the label. Defaults to 'Missing Label'.
        y_label2 (str, optional): Label for 2nd y axis. Ensure that units are
            included in the label. Defaults to 'Missing Label'.
        color1 (str, optional): Color of 1st scatter plot. Defaults to 'red'.
        color2 (str, optional): Color of 2nd scatter plot. Defaults to 'blue'.
        scatter_type (str, optional): Determines the matplotlib.plot parameter
            `fmt` for both scatter plots. Defaults to ''.
        fit (str, optional): Determines which, if any, of the plots should also
            have a line of best fit. Options are ('none', 'both', 'first',
            'second'). Defaults to 'none'.
        month_num (int, optional): Integer representing which month of the year
            to plot year-over-year data for. 1 is January, 2 is February, etc.
            Defaults to 0, which graphs data points for all months together.
            Current implementation only allows graphing of 1 or all months.
    """
    plt.rcParams['figure.figsize'] = [8, 6]

    if month_num != 0:
        data = filter_month(data, month_num)

    fig, ax1 = plt.subplots()

    # Add X axis label
    ax1.set_xlabel(x_label)

    # Add Data, Label, Color, and Point Type to 1 set of Y axis data
    ax1 = plot_single_series(ax1, data[x_data], data[y_data1],
                             y_label1, color1, scatter_type)

    if fit in ['first', 'both']:
        find_best_fit(ax1, data[x_data], data[y_data1], 'orange')

    if y_data2 != '':
        # Adding Twin Axes
        ax2 = ax1.twinx()

        # Add Data, Label, Color, and Point Type to another set of Y axis data
        ax2 = plot_single_series(ax2, data[x_data], data[y_data2],
                                 y_label2, color2, scatter_type)

        if fit in ['second', 'both']:
            find_best_fit(ax2, data[x_data], data[y_data2], 'green')

    # Show plot
    plt.show()
