"""
Loops through list of cities, scrapes data for each city
Packs each set of data into DataFrame
Cleans off specified number of rows at beginning of DataFrame
Saves DataFrame into correspondingly-named CSV in /data directory
"""

from helpers import get_dataframe, clean_data
from constants import DEFAULT_URL, CITIES

# Loop through all city names
for item in CITIES.items():
    # Skip unnecessary test datasets
    if 'clean data test' in item[0]:
        continue

    # Create URL for current city
    wikiurl = f"{DEFAULT_URL}{item[0]}.tab"

    # Using helper function, scrape data from URL and pack into DataFrame
    dataframe = get_dataframe(wikiurl)

    # Drop unnecessary starting rows
    dataframe = clean_data(dataframe, item[0])

    # Save DataFrame into a CSV, systematically-named
    dataframe.to_csv(f'data/{item[1][0]}_weather.csv', index=False)
