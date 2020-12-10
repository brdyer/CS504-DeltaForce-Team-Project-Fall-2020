#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import os


os.chdir("D:/MS in DAEN/GMU 2020/CS 504/Group Delta/Flight Data")


df = pd.read_csv('D:/MS in DAEN/GMU 2020/CS 504/Group Delta/Flight Data/flight_data_4_airlines_12_departure_airports.csv')


flight = df[['YEAR',
        'MONTH',
        'DAY_OF_MONTH',
        'DAY_OF_WEEK',
        'FL_DATE',
        'OP_UNIQUE_CARRIER',
        'ORIGIN',
        'ORIGIN_CITY_NAME',
        'ORIGIN_STATE_ABR',
        'ORIGIN_STATE_NM',
        'CRS_DEP_TIME',
        'DEP_TIME',
        'DEP_DELAY',
        'DEP_DELAY_NEW',
        'DEP_DEL15',
        'DEP_DELAY_GROUP',
        'CANCELLED',
        'DIVERTED',
        'CARRIER_DELAY',
        'WEATHER_DELAY',
        'NAS_DELAY',
        'SECURITY_DELAY',
        'LATE_AIRCRAFT_DELAY']]
print(flight)


# Show diverted flight records
print(flight[flight['DIVERTED'] == 1])


# Drop all diverted flight records
flight.drop(flight.loc[flight['DIVERTED'] == 1].index, inplace=True)
print(flight)


#Show delayed flights caused by weather-nonrelated reasons
print(flight[(flight['DEP_DELAY'] > 0) & (flight['WEATHER_DELAY'] == 0)])


#Drop all records of delayed flights caused by weather-nonrelated reasons
flight.drop(flight.loc[(flight['DEP_DELAY'] > 0) & (flight['WEATHER_DELAY'] == 0)].index, inplace=True)
print(flight)


#Drop columns: CARRIER_DELAY, NAS_DELAY, SECURITY_DELAY, LATE_AIRCRAFT_DELAY, DIVERTED
flight.drop(['CARRIER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DIVERTED'], axis=1, inplace=True)


#Clean all rows contained null values in column DEP_TIME
flight.drop(flight.loc[flight['DEP_TIME'].isnull()].index, inplace=True)


#Replace null values by 0 in WEATHER_DELAY
flight['WEATHER_DELAY'].fillna(0, inplace=True)

flight.isnull().sum()


#Format all values in column CRS_DEP_TIME with 4 digits
flight['CRS_DEP_TIME'] = ["{:04d}".format(i) for i in flight['CRS_DEP_TIME']]


#Covert values in CRS_DEP_TIME to datetime datatype
df = flight['CRS_DEP_TIME']
hour = df.str[:-2]
minutes = df.str[-2:]
time = pd.concat([hour, minutes], axis=1)
time.columns =['hour', 'minutes']
time['time'] = (pd.to_datetime(time['hour'].astype(str) + ':' + time['minutes'].astype(str), format='%H:%M').dt.time)
flight['CRS_DEP_TIME'] = time['time']


#Fuse the time data in CRS_DEP_TIME to FL_DATE 
flight['FL_DATE'] = pd.to_datetime(flight.FL_DATE.astype(str) + ' ' + flight.CRS_DEP_TIME.astype(str))


#Sort data by chronological order
flight.sort_values(by=['FL_DATE'], inplace=True, ascending=True)


#Save to csv file
flight.to_csv('cleaned_flight_data.csv', sep=',', encoding='utf-8', index=False)



