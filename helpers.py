import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
# import icecream as ic
from bs4 import BeautifulSoup

plt.rcParams['figure.figsize'] = [12, 6]

label_dict = {
    'date': ['Time', 'Date'],
    'highTemp': ['Highest Temp', 'Highest Temp (C)'],
    'avgHighTemp': ['Average High Temp', 'Average High Temp (C)'],
    'avgLowTemp': ['Average Low Temp', 'Average Low Temp (C)'],
    'lowTemp': ['Lowest Temp', 'Lowest Temp (C)'],
    'snowfall': ['Snowfall Amount', 'Snowfall (mm)'],
    'snowfallDays': ['Snowfall Days', 'Snowfall Days']
}

month_dict = {
    0: 'All Months',
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}


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


def find_best_fit(ax, x_dataset, y_dataset, color, fit_degree, line_type='-'):
    """_summary_

    Args:
        ax (Axes): Axes object to plot the line of best fit on.
        x_dataset (Series): Series of 1 set of x axis data.
        y_dataset (Series): Series of 1 set of y axis data.
        color (str): Color of best fit line.
        line_type (str, optional): Style of line for the line of best fit.
            Defaults to '-'.
    """
    z = np.polyfit(x_dataset.values.flatten(),
                   y_dataset.values.flatten(), fit_degree)
    p = np.poly1d(z)

    # print(z)

    # coeff = z[0]

    # ### Residual or Sum of Square Error (SSE)
    # SSE = z[1]

    # ### Determining the Sum of Square Total (SST)
    # ## the squared differences between the observed dependent variable and its mean
    # diff = y_dataset.values.flatten() - y_dataset.values.flatten().mean()
    # square_diff = diff ** 2
    # SST = square_diff.sum()

    # ###  Now getting the coefficient of determination (R2)
    # R2 = 1 - SSE/SST
    # print(R2)

    ax.plot(x_dataset, p(x_dataset), line_type, color=color)


def plot_bar_decade(data):

    decade_totals = [0]

    decades = [1891, 1901, 1911, 1921, 1931, 1941,
               1951, 1961, 1971, 1981, 1991, 2001, 2011]

    current_end_decade = 1901

    current_decade_index = 0

    for i, row in data.iterrows():
        # print((row['date'] < current_end_decade).values[0])
        if (row['date'] >= 2021).values[0]:
            break

        if (row['date'] < current_end_decade).values[0]:
            decade_totals[current_decade_index] += row['snowfall'].values[0]
        else:
            decade_totals.append(row['snowfall'].values[0])
            current_decade_index += 1
            current_end_decade += 10

    print(decades, decade_totals)

    plt.bar(decades, decade_totals)

    plt.show()


def plot_in_between(data, x_data, y_data1, y_data2='', color1='red',
                    color2='blue', month_num=0, fit_degree=1):

    if month_num != 0:
        data = filter_month(data, month_num)

    fig, ax1 = plt.subplots()

    x_label = label_dict[x_data][1]
    y_label1 = label_dict[y_data1][1]

    # Add X axis label
    ax1.set_xlabel(x_label)

    z1 = np.polyfit(data[x_data].values.flatten(),
                    data[y_data1].values.flatten(), fit_degree)
    p1 = np.poly1d(z1)

    fit1_flat = [item for sublist in p1(data[x_data]) for item in sublist]

    ax1.plot(data[x_data], fit1_flat, '-', color=color1)

    # y_titles = label_dict[y_data1][0]

    # y_titles = y_titles + " and " + label_dict[y_data2][0]

    # y_label2 = label_dict[y_data2][1]

    # Adding Twin Axes
    # ax2 = ax1.twinx()

    z2 = np.polyfit(data[x_data].values.flatten(),
                    data[y_data2].values.flatten(), fit_degree)
    p2 = np.poly1d(z2)

    fit2_flat = [item for sublist in p2(data[x_data]) for item in sublist]

    ax1.plot(data[x_data], fit2_flat, '-', color=color2)

    ax1.fill_between(data[x_data].values.flatten(),
                     fit1_flat, fit2_flat, alpha=.5, linewidth=0)

    ax1.plot(data[x_data], np.subtract(fit1_flat, fit2_flat), '-', color='purple')

    # graph_title = f"{y_titles} v. {label_dict[x_data][0]} for {month_dict[month_num]}"

    # plt.title(graph_title)

    # Show plot
    plt.show()


def plot_double_scatter(data, x_data, y_data1, y_data2='', color1='red',
                        color2='blue', scatter_type='', fit='none',
                        month_num=0, fit_degree=1):
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
    if month_num != 0:
        data = filter_month(data, month_num)

    fig, ax1 = plt.subplots()

    x_label = label_dict[x_data][1]
    y_label1 = label_dict[y_data1][1]

    # Add X axis label
    ax1.set_xlabel(x_label)

    # Add Data, Label, Color, and Point Type to 1 set of Y axis data
    ax1 = plot_single_series(ax1, data[x_data], data[y_data1],
                             y_label1, color1, scatter_type)

    if fit in ['first', 'both']:
        find_best_fit(ax1, data[x_data], data[y_data1], 'orange', fit_degree)

    y_titles = label_dict[y_data1][0]

    if y_data2 != '':

        y_titles = y_titles + " and " + label_dict[y_data2][0]

        y_label2 = label_dict[y_data2][1]

        # Adding Twin Axes
        ax2 = ax1.twinx()

        # Add Data, Label, Color, and Point Type to another set of Y axis data
        ax2 = plot_single_series(ax2, data[x_data], data[y_data2],
                                 y_label2, color2, scatter_type)

        if fit in ['second', 'both']:
            find_best_fit(ax2, data[x_data],
                          data[y_data2], 'green', fit_degree)

    graph_title = f"{y_titles} v. {label_dict[x_data][0]} for {month_dict[month_num]}"

    plt.title(graph_title)

    # Show plot
    plt.show()
