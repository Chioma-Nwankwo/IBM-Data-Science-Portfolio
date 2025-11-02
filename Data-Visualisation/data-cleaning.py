import pandas as pd
import seaborn as sns
from pandas import DataFrame as df
import numpy as np
import matplotlib.pyplot as plt
import sys
import re

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 50)
url = r"C:\Users\chiom\Downloads\jfk_weather.csv"
raw_data = pd.read_csv(url, parse_dates=['DATE'], low_memory=False)
raw_data.head()
raw_data.dtypes
# Choose what columns to import from raw data
column_subset = [
    'DATE',
    'HOURLYVISIBILITY',
    'HOURLYDRYBULBTEMPF',
    'HOURLYWETBULBTEMPF',
    'HOURLYDewPointTempF',
    'HOURLYRelativeHumidity',
    'HOURLYWindSpeed',
    'HOURLYWindDirection',
    'HOURLYStationPressure',
    'HOURLYPressureTendency',
    'HOURLYSeaLevelPressure',
    'HOURLYPrecip',
    'HOURLYAltimeterSetting'
]

# Filter dataset to relevant columns
hourly_data = raw_data[column_subset]
# Set date index
hourly_data = hourly_data.set_index(pd.DatetimeIndex(hourly_data['DATE']))
#Remove DATE column
hourly_data.drop(['DATE'], axis=1, inplace=True)
# Convert to numeric, forcing errors to NaN
hourly_data.replace(to_replace='*', value=np.nan, inplace=True)
# Handle missing values, We can see that some values end with an s (indicating snow), while there is a strange value 0.020.01s which appears to be an error of some sort. To deal with T values, we will set the observation to be 0. We will also replace the erroneous value 0.020.01s with NaN.
hourly_data['HOURLYPrecip'].replace(to_replace='T', value='0.00', inplace=True)
hourly_data['HOURLYPrecip'].replace('0.020.01s', np.nan, inplace=True)
# Set of columns to convert, skips the DATE column which is already numeric and focuses on the remaining columns which have some non-numeric values
messy_columns = column_subset[1:]
# Convert columns to float32 datatype
for i in messy_columns:
    hourly_data[i] = hourly_data[i].apply(lambda x: re.sub('[^0-9,.-]', '', x) if type(x) == str else x).replace('', np.nan).astype(('float32'))

hourly_data.describe()
#inspect unique values in HOURLYPressureTendency column
hourly_data['HOURLYPressureTendency'].unique()
# Check that values in HOURLYPressureTendency are between 0 and 8
# If there are any values outside this range, cond will be non-zero
cond = len(hourly_data[~hourly_data['HOURLYPressureTendency'].isin(list(range(0,9)) + [np.nan])])
'Hourly Pressure Tendency should be between 0 and 8: {}'.format(cond == 0)
# Replace any hourly visibility figure outside these bounds with nan
hourly_data.loc[hourly_data['HOURLYVISIBILITY'] > 10, 'HOURLYVISIBILITY'] = np.nan
# Check that values in HOURLYVISIBILITY are between 0 and 10 miles
# If there are any values outside this range, cond will be non-zero
cond= len(hourly_data[(hourly_data['HOURLYVISIBILITY'] < 0) | (hourly_data['HOURLYVISIBILITY'] > 10)])
'Hourly Visibility should be between 0 and 10: {}'.format(cond == 0)
hourly_data['HOURLYVISIBILITY'].describe()
hourly_data['HOURLYVISIBILITY'].unique()
# Check for duplicate date entries
# If there are any duplicates, cond will be non-zero
cond = len(hourly_data[hourly_data.index.duplicated()].sort_index())
'Date index contains no duplicate entries: {}'.format(cond == 0)
#Reset index to turn the date index back into a column
hourly_data.reset_index(inplace=True)
#fill missing values for HOURLYPressureTendency and interpolate other missing values
#HOURLYPressureTendency is a categorical variable, so we will fill missing values with the last valid observation
hourly_data['HOURLYPressureTendency'] = hourly_data['HOURLYPressureTendency'].fillna(method='ffill') # fill with last valid observation
hourly_data = hourly_data.interpolate(method='linear') # interpolate missing values
hourly_data.drop(hourly_data.index[0], inplace=True) # drop first row which still has a missing value
# Transform HOURLYWindDirection into a cyclical variable using sin and cos transforms
hourly_data['HOURLYWindDirectionSin'] = np.sin(hourly_data['HOURLYWindDirection']*(2.*np.pi/360))
hourly_data['HOURLYWindDirectionCos'] = np.cos(hourly_data['HOURLYWindDirection']*(2.*np.pi/360))
#drops the original HOURLYWindDirection column
hourly_data.drop(['HOURLYWindDirection'], axis=1, inplace=True)
# Transform HOURLYPressureTendency into 3 dummy variables based on NOAA documentation
hourly_data['HOURLYPressureTendencyIncr'] = [1.0 if x in [0,1,2,3] else 0.0 for x in hourly_data['HOURLYPressureTendency']] # 0 through 3 indicates an increase in pressure over previous 3 hours
hourly_data['HOURLYPressureTendencyDecr'] = [1.0 if x in [5,6,7,8] else 0.0 for x in hourly_data['HOURLYPressureTendency']] # 5 through 8 indicates a decrease over previous 3 hours
hourly_data['HOURLYPressureTendencyConst'] = [1.0 if x == 4 else 0.0 for x in hourly_data['HOURLYPressureTendency']] # 4 indicates no change during previous 3 hours
hourly_data.drop(['HOURLYPressureTendency'], axis=1, inplace=True)
hourly_data['HOURLYPressureTendencyIncr'] = hourly_data['HOURLYPressureTendencyIncr'].astype(('float32'))
hourly_data['HOURLYPressureTendencyDecr'] = hourly_data['HOURLYPressureTendencyDecr'].astype(('float32'))
hourly_data['HOURLYPressureTendencyConst'] = hourly_data['HOURLYPressureTendencyConst'].astype(('float32'))
# define the new column names
columns_new_name = [
    'DATE',
    'visibility',
    'dry_bulb_temp_f',
    'wet_bulb_temp_f',
    'dew_point_temp_f',
    'relative_humidity',
    'wind_speed',
    'station_pressure',
    'sea_level_pressure',
    'precip',
    'altimeter_setting',
    'wind_direction_sin',
    'wind_direction_cos',
    'pressure_tendency_incr',
    'pressure_tendency_decr',
    'pressure_tendency_const'
]
# create a mapping of old column names to new column names
columns_name_map = {c:columns_new_name[i] for i, c in enumerate(hourly_data.columns)}

hourly_data_renamed = hourly_data.rename(columns=columns_name_map)
# Set date index
hourly_data_renamed['DATE'] = pd.to_datetime(hourly_data_renamed['DATE'])
hourly_data_renamed.set_index('DATE', inplace=True)
# Inspect the final dataset
print(hourly_data_renamed.info())
hourly_data_renamed.head()
# Explore some general information about the dataset
print('# of megabytes held by dataframe: ' + str(round(sys.getsizeof(hourly_data_renamed) / 1000000,2)))
# Number of columns 
print('# of features: ' + str(hourly_data_renamed.shape[1])) 
#Number of rows
print('# of observations: ' + str(hourly_data_renamed.shape[0]))
print('Start date: ' + str(hourly_data_renamed.index[0]))
print('End date: ' + str(hourly_data_renamed.index[-1]))
print('# of days: ' + str((hourly_data_renamed.index[-1] - hourly_data_renamed.index[0]).days))
print('# of months: ' + str(round((hourly_data_renamed.index[-1] - hourly_data_renamed.index[0]).days/30,2)))
print('# of years: ' + str(round((hourly_data_renamed.index[-1] - hourly_data_renamed.index[0]).days/365,2)))
hourly_data_renamed.to_csv("jfk_weather_cleaned.csv", float_format='%g')
print(hourly_data_renamed)
