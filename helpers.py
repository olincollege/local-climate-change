"""
All helper functions stored in this file.
"""

import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup
from numpy.polynomial import Polynomial as P

# Import from our constants file
from constants import CITIES, LABEL_DICT, MONTH_DICT

# Set size of graphs [width, height]
plt.rcParams['figure.figsize'] = [12, 6]


def convert_year(input_year_series):
    """
    In our data, the dates come in the format '2022-03', but we want them
    in the form of floats so we can do calculations and graph them. This
    function converts an incoming Series of dates into floats and returns a
    list of float dates.

    Args:
        input_year_series (Series): Series of string dates

    Returns:
        list: List of float dates
    """
    added_list = []
    for item in input_year_series:

        # Convert month number 1-12 into decimal (fraction over 12)
        month = float(item[1]) / 12

        # Make year into its own object
        year = item[0]

        # Append year + month decimal to list
        added_list.append(float(year) + month)

    return added_list


def get_dataframe(wikiurl):
    """
    Takes WikiMedia URL of weather dataset, returns the datatable from the
    webpage in the form of a DataFrame.

    Args:
        wikiurl (str): Full URL for WikiMedia page with dataset for particular
            city

    Returns:
        DataFrame: Web table data packed into DataFrame
    """

    # get the response in the form of html
    response = requests.get(wikiurl)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    weather_table = soup.find('table', {'class': "mw-tabular"})

    data = pd.read_html(str(weather_table))

    # convert list to dataframe
    data = pd.DataFrame(data[0])

    return data


def clean_data(dataframe, city):
    """
    Takes DataFrame for one city's weather data, removes specified number of
    rows from beginning of DataFrame, converts date values from hyphenated
    strings to fractional floats, and returns cleaned DataFrame.

    Args:
        dataframe (DataFrame): raw set of data for one city
        city (str): name of city corresponding to current dataset, used to
            find number of starting rows to drop.

    Returns:
        DataFrame: Returns DataFrame after initial rows have been dropped and
            dates reformatted.
    """

    # Remove first X rows and reindex because some cities have a bunch of data
    # missing in the first few years.
    dataframe.drop(range(CITIES[city][1]), axis=0, inplace=True)

    # Create Series of just the dates datapoints
    dates_series = pd.Series(dataframe.date.values.flatten())

    # Using helper function, replace dates column with float-ified dates
    dataframe['date'] = convert_year(dates_series.str.split('-'))

    return dataframe


def compile_csvs():
    """
    Loops through all of the listed cities and reads from each of the
    corresponding CSVs. Saves each of these DataFrames into a dictionary.

    Args:
        None

    Returns:
        dict: Returns dictionary with city names as keys and DataFrames as
            values.
    """

    # Creates empty dictionary for DataFrames to be added to.
    df_dict = {}

    # Loops through cities
    for item in CITIES.items():
        # do not include clean data unit tests
        if 'clean data test' in item[0]:
            continue
        # For each city, pull from CSV, save to DataFrame (while skipping
        # extra header rows) and save into dictionary.
        df_dict[item[0]] = pd.read_csv(
            f'data/{item[1][0]}_weather.csv', skiprows=[1, 2])

    # Return this new dictionary of DataFrames
    return df_dict


def plot_single_series(axes, x_data, y_data, color, scatter_type):
    """
    Takes one set of X data and one set of Y data and plots a scatter plot on
    the given Axes, returning the modified Axes.

    Args:
        axes (Axes): Axes object to add scatter plot to
        x_data (Series): Dataset of X points
        y_data (Series): Dataset of Y points
        color (str): Color name for scatter plot and axis labels
        scatter_type (str): Scatter plot line type for plt.plot ('.', '-', etc)

    Returns:
        Axes: Return modified Axes with scatter plot added
    """

    # Set Y Label and Y Label Color based on parameters
    axes.set_ylabel(LABEL_DICT[y_data.name][1], color=color)

    # Plot single set of data
    axes.plot(x_data, y_data, scatter_type, color=color, alpha=0.4)

    # Add axis tick marks
    axes.tick_params(axis='y', labelcolor=color)

    return axes


def filter_month(data, month_num):
    """
    Takes DataFrame and number of month to filter data for. Drops all rows
    from DataFrame except for those of the requested month.

    Args:
        data (DataFrame): DataFrame of all data for current city.
        month_num (int): Number of month to filter data for (1 = January, 12 =
            December, etc)

    Returns:
        DataFrame: Returns DataFrame consisting of only data for requested month
    """
    # Determine length of DataFrame (# of rows)
    series_length = int(data.shape[0])

    # Loop through each month. If the current loop is not equal to the month the
    # filter is looking for, drop the row of data.
    for drop_index in range(1, 13):

        # Don't drop if the loop is on the requested month
        if drop_index == month_num:
            continue

        # Aggregate all rows of current month
        range_to_drop = range(drop_index - 1, series_length, 12)

        # Drop all rows of current month
        data.drop(range_to_drop, axis=0, inplace=True)

    return data


def find_best_fit(data, x_data, y_data, fit_degree):
    """
    Find line of best fit of a set of X and Y data. Can also take a degree
    of polynomial fit.

    Args:
        data (DataFrame): DataFrame object to calculate line of best fit from.
        x_data (str): Column header for x data
        y_data (str): Column header for y data
        fit_degree (int): Degree of polynomial fit.

    Returns:
        poly (Series): Series containing polynomial fit equation
    """

    # Drop any rows with NaN values in the columns we are using, to not upset
    # the fit function.
    data.dropna(axis='rows', subset=[x_data, y_data], inplace=True)

    x_dataset = data[x_data]
    y_dataset = data[y_data]

    # Find polynomial fit
    poly = P.fit(x_dataset.values.flatten(),
                 y_dataset.values.flatten(), fit_degree)

    return poly


def plot_bar_decade(data, date_range, y_data):
    """
    For a certain category of Y data and a certain date range, plot a bar
    graph of total values by decade. For example, graphs sum of snowfall for
    each decade from 1891 to 2021.

    Args:
        data (DataFrame): DataFrame of all weather data for specific city.
        date_range (List): Start year to end year in the form of a list. Comes
            in the form [1891, 2021] an(d will start with the decade beginning
            1891 and end with the decade ending 2020.
        y_data (str): Column header for Y dataset
    """

    # Empty list of decade sum values
    decade_totals = [0]

    # Creates list of decade years
    decades = range(date_range[0], date_range[1], 10)

    current_end_decade = decades[1]

    current_decade_index = 0

    # Loops through DataFrame rows
    for _, row in data.iterrows():

        # If the date is past the end year, break the loop
        if row['date'] >= date_range[1]:
            break

        # If the date is still within the correct decade
        if row['date'] < current_end_decade:

            # If the current datapoint is NaN, print error message
            if np.isnan(row[y_data]):
                print(f"Warning: Found NaN value at {row['date']}")

            else:
                # Add the data to the current decade's running total
                decade_totals[current_decade_index] += row[y_data]

        # If new decade, append current datapoint to create new decade sum
        else:
            decade_totals.append(row[y_data])
            current_decade_index += 1
            current_end_decade += 10

    # Add axis labels and title
    plt.xlabel(LABEL_DICT['date'][1])
    plt.ylabel(f"Total {LABEL_DICT[y_data][1]}")

    plt.title(f"Cumulative {LABEL_DICT[y_data][0]} by Decade")

    # Create bar plot
    plt.bar(decades, decade_totals)

    plt.show()


def plot_in_between(data, x_data, y_data, month_num=0, fit_degree=1):
    """
    Graphs lines of best fit for 2 datasets, fills area between these two
    lines.

    Args:
        data (DataFrame): DataFrame of weather data.
        x_data (str): DataFrame column label for x dataset. Expects 'date' or
            or 'avgLowTemp' format.
        y_data (list): List containing DataFrame column labels for 1st and 2nd
            y datasets. Expects ['date', 'avgLowTemp'] format.
        month_num (int, optional): Integer representing which month of the year
            to plot year-over-year data for. 1 is January, 2 is February, etc.
            Defaults to 0, which graphs lines for all months together.
            Current implementation only allows graphing of 1 or all months.
        fit_degree (int, optional): Integer representing degree of polynomial
            fit for lines of best fit. Defaults to 1, signifying a linear fit.
    """

    # If month_num is 0, all months should be plotted. Otherwise, filter data
    # to drop everything except the requested month.
    if month_num != 0:
        data = filter_month(data, month_num)

    ax1 = plt.subplots()[1]

    y_data1 = y_data[0]
    y_data2 = y_data[1]

    # Set X label to stylized version of the column header
    x_label = LABEL_DICT[x_data][1]
    y_label1 = LABEL_DICT[y_data1][1]
    y_label2 = LABEL_DICT[y_data2][1]

    # Add X axis label
    ax1.set_xlabel(x_label)

    # Find fit line of 1st dataset
    para_1 = find_best_fit(data, x_data, y_data1, fit_degree)

    # Flatten fit line for plotting
    para_1 = para_1(data[x_data]).values.flatten()

    # Plot fit of 1st dataset
    ax1.plot(data[x_data], para_1, '-', color='red')

    # Set Y axis label
    ax1.set_ylabel(f"{y_label1} and {y_label2}")

    # Find fit line of 2nd dataset
    para_2 = find_best_fit(data, x_data, y_data2, fit_degree)

    # Flatten fit line for plotting
    para_2 = para_2(data[x_data]).values.flatten()

    # Plot fit for 2nd element
    ax1.plot(data[x_data], para_2, '-', color='blue')

    # Fill between both fit lines.
    ax1.fill_between(data[x_data].values.flatten(),
                     para_1, para_2, alpha=.5, linewidth=0)

    # Concatenate both Y titles
    y_titles = LABEL_DICT[y_data1][0] + " and " + LABEL_DICT[y_data2][0]

    # Create full graph title
    graph_title = f"{y_titles} v. {LABEL_DICT[x_data][0]} for {MONTH_DICT[month_num]}"

    # Add title
    plt.title(graph_title)

    # Show plot
    plt.show()


def plot_double_scatter(data_list, y_data2='', fit='none',
                        month_num=0, fit_degree=1):
    """
    Takes a DataFrame of weather data and 1-2 column names, along with some
    other optional arguments, and plots the given columns. Can also create lines
    of best fit for the plots.

    Args:
        data_list (list): List containing multiple required arguments:
            data (DataFrame): DataFrame of weather data.
            x_data (str): DataFrame column label for x dataset. Expects 'date'
                or 'avgLowTemp' format.
            y_data1 (str): DataFrame column label for 1st y dataset. Expects
                'date' or 'avgLowTemp' format.
        y_data2 (str, optional): DataFrame column label for 2nd y dataset.
            Expects 'date' or 'avgLowTemp' format. Defaults to '', wherein the
            function only plots 1 dataset.
        fit (str, optional): Determines which, if any, of the plots should also
            have a line of best fit. Options are ('none', 'both', 'first',
            'second'). Defaults to 'none'.
        month_num (int, optional): Integer representing which month of the year
            to plot year-over-year data for. 1 is January, 2 is February, etc.
            Defaults to 0, which graphs data points for all months together.
            Current implementation only allows graphing of 1 or all months.
        fit_degree (int, optional): Integer representing degree of polynomial
            fit for lines of best fit. Defaults to 1, signifying a linear fit.
    """
    # Separate arguments from list
    data = data_list[0]
    x_data = data_list[1]
    y_data1 = data_list[2]

    # If month_num is 0, all months should be plotted. Otherwise, filter data
    # to drop everything except the requested month.
    if month_num != 0:
        data = filter_month(data, month_num)

    ax1 = plt.subplots()[1]

    # Add X axis label
    ax1.set_xlabel(LABEL_DICT[x_data][1])

    # Add Data, Label, Color, and Point Type to 1 set of Y axis data
    ax1 = plot_single_series(ax1, data[x_data], data[y_data1], 'red', '')

    # Find best fit of first dataset if requested
    if fit in ['first', 'both']:
        para_1 = find_best_fit(data, x_data, y_data1, fit_degree)

        # Plot line of best fit
        ax1.plot(data[x_data], para_1(data[x_data]), '-', color='red')

    # Add stylized title for 1st Y axis
    y_titles = LABEL_DICT[y_data1][0]

    # If 2nd dataset present
    if y_data2 != '':

        # Add stylized title for 2nd Y axis
        y_titles = y_titles + " and " + LABEL_DICT[y_data2][0]

        # Adding Twin Axes
        ax2 = ax1.twinx()

        # Add Data, Label, Color, and Point Type to another set of Y axis data
        ax2 = plot_single_series(ax2, data[x_data], data[y_data2],
                                 'blue', '')

        # Find best fit of second dataset if requested
        if fit in ['second', 'both']:
            para_2 = find_best_fit(data, x_data, y_data2, fit_degree)

            # Plot line of best fit
            ax2.plot(data[x_data], para_2(data[x_data]), '-', color='blue')

    # Concatenate graph titles into fully auto-generated graph title
    graph_title = f"{y_titles} v. {LABEL_DICT[x_data][0]} for {MONTH_DICT[month_num]}"

    # Add title to graph
    plt.title(graph_title)

    # Show plot
    plt.show()
