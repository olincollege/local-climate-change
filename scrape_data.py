from helpers import *
from constants import *

# Loops through list of cities, scrapes data for each city
# Packs each set of data into DataFrame
# Cleans off specified number of rows at beginning of DataFrame
# Saves DataFrame into correspondingly-named CSV in /data directory

# Loop through all city names
for city in CITIES:

    # Create URL for current city
    wikiurl = f"{DEFAULT_URL}{city}.tab"

    # Using helper function, scrape data from URL and pack into DataFrame
    dataframe = get_dataframe(wikiurl)

    # Drop unnecessary starting rows
    dataframe = clean_data(dataframe, city)

    # Save DataFrame into a CSV, systematically-named
    dataframe.to_csv(f'data/{CITIES[city][0]}_weather.csv', index=False)
