import pandas as pd
import requests
from bs4 import BeautifulSoup

def convert_year(input_year_list):
    added_list = []
    for i in range(len(input_year_list)):
            month = float(input_year_list[i][1]) / 12
            year = input_year_list[i][0]
            added_list.append(float(year) + month)
    return added_list

def get_dataframe(wikiurl):
    # get the response in the form of html
    table_class="mw-tabular sortable jquery-tablesorter"
    response=requests.get(wikiurl)
    # print(response.status_code)

    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    weather_table=soup.find('table',{'class':"mw-tabular"})

    df=pd.read_html(str(weather_table))
    # convert list to dataframe
    df=pd.DataFrame(df[0])

    return df