
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from helpers import *


wikiurl = "https://commons.wikimedia.org/wiki/Data:Ncei.noaa.gov/weather/Boston.tab"
data = get_dataframe(wikiurl)

# drop the unwanted columns
data.drop(["precip", "precipDays"], axis=1, inplace = True)
data.drop(range(226), axis = 0, inplace = True)


dates_series = pd.Series(data.date.values.flatten())

data['date'] = convert_year(dates_series.str.split('-'))


dates = data['date']
avg_high_temp = data['avgHighTemp']
snowfall = data["snowfall"]




# plt.plot(dates.tail(500), snowfall.tail(500), ".", label = 'Snowfall (mm)', c = 'blue')
# plt.plot(dates.tail(500), high_temp.tail(500), ".", label = 'High Temp (C)', c = 'red')

fig, ax1 = plt.subplots() 
  
ax1.set_xlabel('Year') 
ax1.set_ylabel('Average High Temperature (C)', color = 'red') 
ax1.plot(dates, avg_high_temp, color = 'red') 
ax1.tick_params(axis ='y', labelcolor = 'red') 
  
# Adding Twin Axes

ax2 = ax1.twinx() 
  
ax2.set_ylabel('Snowfall (mm)', color = 'blue') 
ax2.plot(dates, snowfall, color = 'blue') 
ax2.tick_params(axis ='y', labelcolor = 'blue') 


# z = np.polyfit(dates_converted[-500:], snowfall.tail(500), 1)
# print(z)
# p = np.poly1d(z)

# plt.plot(dates_converted[-500:],p(dates_converted[-500:]),"r--")

# plt.xlabel("Year")
# plt.ylabel("Snowfall (mm)")

plt.show()

# plt.regplot(x=dates_converted[-500:],y=snowfall.tail(500), fit_reg=True) 


