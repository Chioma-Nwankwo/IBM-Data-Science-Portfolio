# Exploratory Data Analysis
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import sys
import re
plt.rcParams['figure.dpi'] = 160
url = r"C:\Users\chiom\Documents\IBM Data Science Professional Certificate\jfk_weather_cleaned.csv"
data = pd.read_csv(url, parse_dates=['DATE'])
# Set date index
data.set_index('DATE', inplace=True)  # Set index directly
# Only drop 'DATE' if it still exists as a column
if 'DATE' in data.columns:
    data.drop(['DATE'], axis=1, inplace=True)
#Load the dataset
print(data.info())
# Columns to visualize
plot_cols = ['dry_bulb_temp_f', 'relative_humidity', 'wind_speed', 'station_pressure', 'precip']
plt.figure(figsize=(15,7))
TEMP_COL = 'dry_bulb_temp_f'

# Plot rolling mean of temperature 
plt.subplot(1, 2, 1)
data.loc['2017', TEMP_COL].rolling('5D').mean().plot(zorder=2)  # 5-day rolling mean
data.loc['2017', TEMP_COL].plot(zorder=1)
plt.legend(['Rolling', 'Temp'])
plt.title('Rolling Avg in Hourly Temperature in 2017')
plt.ylabel('Temperature (F)')
# Plot rolling mean of temperature for Jan–Mar 2017
plt.subplot(1, 2, 2)
data.loc['2017-01':'2017-03', TEMP_COL].rolling('2D').mean().plot(zorder=2)  # 2-day rolling mean
data.loc['2017-01':'2017-03', TEMP_COL].plot(zorder=1)
plt.legend(['Rolling', 'Temp'])
plt.title('Rolling Avg in Hourly Temperature in Winter 2017')
plt.ylabel('Temperature (F)')
plt.tight_layout()
plt.show()

# %%
#Import required modules
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import seaborn as sns
from pandas import DataFrame as df
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import sys
import re
plt.rcParams['figure.dpi'] = 160
url = r"C:\Users\chiom\Documents\IBM Data Science Professional Certificate\jfk_weather_cleaned.csv"
data = pd.read_csv(url, parse_dates=['DATE'])
data.reset_index(inplace=True)
data['DATE'] = pd.to_datetime(data['DATE'])

# Step 2: Set 'DATE' as the index
data.set_index('DATE', inplace=True)

# Step 3: Now slice using date strings
sample = data['2016-01-01':'2018-01-01']
sample.info()
def split_data(data, val_size=0.1, test_size=0.1):
    """
    Splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data) * (1 - test_size - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
    
    return df_train, df_val, df_test


# Create data split
df_train, df_val, df_test = split_data(sample)

print('Total data size:      {} rows'.format(len(sample)))
print('Training set size:    {} rows'.format(len(df_train)))
print('Validation set size:  {} rows'.format(len(df_val)))
print('Test set size:        {} rows'.format(len(df_test)))