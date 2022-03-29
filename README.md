# local-climate-change

This repository contains the midterm project for the Software Design course at Olin College of Engineering in Needham, MA.

## Description

We've noticed that the vast majority of global warming and climate change information in the media centers around either record-breaking disasters or complex scientific studies inaccessible to the average citizen. We decided to visualize weather data over time, not in the Arctic or the stratosphere, but from the same local weather stations that supply your weather app the current temperature when you wake up in the morning.

We decided to dig into data collected by the ***National Oceanic and Atmospheric Administration (NOAA)***, more specifically their ***National Centers for Environmental Information (NCEI)***. By looking at these historical weather datasets from a wide variety of cities around the United States, we can begin to visualize the increase in global temperatures and ever-changing global climate.

Our hypothesis is that if humans can see the data behind their anecdotal observations of a warming climate, and if these datasets are representative of their own backyards, they will be more inclined to understand the magnitude and urgency of the climate crisis.



## Building in Linux

This project was developed in a WSL environment running Ubuntu 20.04. Repository setup should be similar when using other operating systems.

This project was developed using the Anaconda Python kernel, and as such our setup instructions will assume Anaconda is being used.



### Dependencies

This project utilizes the following list of dependencies:
- Pandas
    - `conda install -c anaconda pandas`
- Requests
    - `conda install -c conda-forge requests`
- BeautifulSoup4
    - `conda install -c conda-forge beautifulsoup4`
- NumPy
    - `conda install -c conda-forge numpy`
- MatPlotLib
    - `conda install -c conda-forge matplotlib`



### Building the Computational Essay

The Jupyter notebook included in this project includes pre-made visualizations and analysis. To experience this notebook, no datasets need to be downloaded, as the necessary CSVs are already included in the `/data` subdirectory of the repository. To view the computational essay, simply run the notebook itself.


### Utilize Functions

If you are interested in exploring the data for yourself, start by exploring the cities listed in WikiMedia Commons at `https://commons.wikimedia.org/wiki/Data:Ncei.noaa.gov/weather/{city}.tab`.

Then, add entries for your selected cities to the dictionary at `constants.py > CITIES`, of the format `{'City Name For URL', 'city_name_for_file', # of rows to drop}`. Next, run `scrape_data.py` and utilize the helper functions within `helpers.py` to gather, clean, and plot your data.

Reference function docstrings and comments for more specific instruction.



## Authors

Andrew Phillips, Brooke Aspen Mague Moss