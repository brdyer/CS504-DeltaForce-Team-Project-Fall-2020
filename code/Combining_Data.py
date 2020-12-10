#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os
import datetime

os.chdir("D:/MS in DAEN/GMU 2020/CS 504/Group Delta")
flight = pd.read_csv('D:/MS in DAEN/GMU 2020/CS 504/Group Delta/Flight Data/cleaned_flight_data.csv')
weather = pd.read_csv('D:/MS in DAEN/GMU 2020/CS 504/Group Delta/Weather Data/Weather_Data_Complete_Raw_70PercentPopulated.csv')

#Convert to datetime datatype
flight['FL_DATE'] = pd.to_datetime(flight['FL_DATE'])

#Convert to datetime datatype
weather['DATE'] = pd.to_datetime(weather['DATE'])

#Add column AirportID to weather data
weather['AirportID'] = weather['STATION']
weather['AirportID'].replace(to_replace = [72259003927,74486094789,72405013743,72219013874,
                                            72793024233,72406093721,72205012815,72295023174,
                                            72494023234,72403093738,72565003017,72530094846], 
                             value =['DFW','JFK','DCA','ATL','SEA','BWI','MCO','LAX','SFO','IAD','DEN','ORD'], inplace=True)


#Sort weather data by date and time
weather.sort_values(by=['DATE'], inplace=True, ascending=True)


#Get the list of column names from flight dataset
col1 = list(flight.columns)


#Get the list of column names from weather dataset
col2 = list(weather.columns)


#Combine 2 lists of column names
col = col1+col2
print(col)


list = ['DFW','JFK','DCA','ATL','SEA','BWI','MCO','LAX','SFO','IAD','DEN','ORD']


#Create empty dataset with all the columns from flight and weather datasets
combine = pd.DataFrame(columns = col)


#Integrate flight and weather data
for x in range(0,len(list)):
    df1 = weather[weather['AirportID'] == list[x]]
    df2 = flight[flight['ORIGIN'] == list[x]]
    df = pd.merge_asof(df2, df1, 
                        left_on='FL_DATE',
                        right_on='DATE',
                        tolerance=pd.Timedelta(hours=1),
                        direction='backward')
    combine = pd.concat([combine,df])


combine.to_csv('combine_data.csv', sep =',', encoding ='utf-8', index=False)

