from helpers import *
from constants import *

# Loop through all city names
for city in CITIES:

    # Create URL for current city
    wikiurl = f"{DEFAULT_URL}{city}.tab"

    # Using helper function, scrape data from URL and pack into DataFrame
    dataframe = get_dataframe(wikiurl)

    # Remove first X rows and reindex because some cities have a bunch of data
    # missing in the first few years.
    dataframe.drop(range(CITIES[city][1]), axis=0, inplace=True)
    dataframe.reset_index(drop=True, inplace=True)

    # Create Series of just the dates datapoints
    dates_series = pd.Series(dataframe.date.values.flatten())

    # Using helper function, replace dates column with float-ified dates
    dataframe['date'] = convert_year(dates_series.str.split('-'))

    # Save DataFrame into a CSV, systematically-named
    dataframe.to_csv(f'data/{CITIES[city][0]}_weather.csv', index=False)
