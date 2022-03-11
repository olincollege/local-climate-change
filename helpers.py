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


def plot_double_scatter(x_data, y_data1, y_data2, x_label='Year',
                        y_label1='Missing Label', y_label2='Missing Label',
                        color1='red', color2='blue'):
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

    point_type = '.'
    fig, ax1 = plt.subplots()

    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label1, color=color1)
    ax1.plot(x_data, y_data1, point_type, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Adding Twin Axes

    ax2 = ax1.twinx()

    ax2.set_ylabel(y_label2, color=color2)
    ax2.plot(x_data, y_data2, point_type, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # z = np.polyfit(x_data.values.flatten(), y_data1.values.flatten(), 1)
    # print(z)
    # p = np.poly1d(z)

    # plt.plot(x_data, p(x_data), "o", color='yellow')

    plt.show()
