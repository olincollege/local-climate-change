import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup


def convert_year(input_year_list):
    """_summary_

    Args:
        input_year_list (_type_): _description_

    Returns:
        _type_: _description_
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
        wikiurl (_type_): _description_

    Returns:
        _type_: _description_
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
        ax (_type_): _description_
        x_data (_type_): _description_
        y_data (_type_): _description_
        y_label (_type_): _description_
        color (_type_): _description_
        scatter_type (_type_): _description_

    Returns:
        _type_: _description_
    """    
    ax.set_ylabel(y_label, color=color)
    ax.plot(x_data, y_data, scatter_type, color=color)
    ax.tick_params(axis='y', labelcolor=color)
    return ax


def find_best_fit(ax, x_data, y_data, color, line_type='-'):
    """_summary_

    Args:
        ax (_type_): _description_
        x_data (_type_): _description_
        y_data (_type_): _description_
        color (_type_): _description_
        line_type (str, optional): _description_. Defaults to '-'.
    """    
    z = np.polyfit(x_data.values.flatten(), y_data.values.flatten(), 1)
    p = np.poly1d(z)

    ax.plot(x_data, p(x_data), line_type, color=color)


def plot_double_scatter(x_data, y_data1, y_data2, x_label='Year', \
                        y_label1='Missing Label', y_label2='Missing Label', \
                        color1='red', color2='blue', scatter_type='', \
                        fit='none'):
    """_summary_

    Args:
        dates (_type_): _description_
        y_data1 (_type_): _description_
        y_data2 (_type_): _description_
        x_label (str, optional): _description_. Defaults to 'Year'.
        y_label1 (str, optional): _description_. Defaults to 'Missing Label'.
        y_label2 (str, optional): _description_. Defaults to 'Missing Label'.
        color1 (str, optional): _description_. Defaults to 'red'.
        color2 (str, optional): _description_. Defaults to 'blue'.
    """

    fig, ax1 = plt.subplots()

    # Add X axis label
    ax1.set_xlabel(x_label)

    # Add Data, Label, Color, and Point Type to 1 set of Y axis data
    ax1 = plot_single_series(ax1, x_data, y_data1,
                             y_label1, color1, scatter_type)

    if fit in ['first', 'both']:
        find_best_fit(ax1, x_data, y_data1, 'yellow')

    # Adding Twin Axes
    ax2 = ax1.twinx()

    # Add Data, Label, Color, and Point Type to another set of Y axis data
    ax2 = plot_single_series(ax2, x_data, y_data2,
                             y_label2, color2, scatter_type)
    
    if fit in ['second', 'both']:
        find_best_fit(ax2, x_data, y_data2, 'green')

    # Show plot
    plt.show()
