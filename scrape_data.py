from helpers import *
from constants import *

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
