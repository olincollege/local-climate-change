# URL = DEFAULT_URL + `City Name` + '.tab'
DEFAULT_URL = \
    'https://commons.wikimedia.org/wiki/Data:Ncei.noaa.gov/weather/'

# 'Stylized City Name': ['simplified_city_name', # of rows to drop]
CITIES = {
    'Boston': ['boston', 226],
    'New York City': ['new_york_city', 12],
    'Chicago': ['chicago', 150],
    'Denver': ['denver', 120],
    'Bismarck, North Dakota': ['bismarck', 135],
    'Detroit': ['detroit', 132],
    # 'Atlanta': ['atlanta', ],
    'Anchorage, Alaska': ['anchorage', 2],
    # 'London': ['london', 1],
    'Minneapolis': ['minneapolis', 147],
    'Temperature Experiment': ['test', 0]
}

# 'columnHeader': ['For Graph Title', 'For Graph Axis Labels']
LABEL_DICT = {
    'date': ['Time', 'Date'],
    'highTemp': ['Highest Temp', 'Highest Temp (C)'],
    'avgHighTemp': ['Average High Temp', 'Average High Temp (C)'],
    'avgLowTemp': ['Average Low Temp', 'Average Low Temp (C)'],
    'lowTemp': ['Lowest Temp', 'Lowest Temp (C)'],
    'precip': ['Precipitation', 'Precipitation (mm)'],
    'precipDays': ['Precipitation Days', 'Precipitation Days'],
    'snowfall': ['Snowfall Amount', 'Snowfall (mm)'],
    'snowfallDays': ['Snowfall Days', 'Snowfall Days']
}

# Month #: 'Month Name'
MONTH_DICT = {
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
