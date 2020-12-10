#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import glob

#Change to your desired working directory
os.chdir("D:/MS in DAEN/GMU 2020/CS 504/Group Delta/Flight Data")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]


#Sort all the file names in numerical and alphabetical order
all_filenames.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
print(all_filenames)


#Combine all files in the list
flight_data = pd.concat([pd.read_csv(f) for f in all_filenames])


list(flight_data.columns)


flight_data.head()


#Filter the dataset with only 4 airlines: AA, DL, WN, UA
df = flight_data.loc[flight_data['OP_UNIQUE_CARRIER'].isin(['AA','DL','WN','UA'])]


df.to_csv( "flight_data_AA_DL_WN_UA.csv", index=False, sep=',', encoding='utf-8')

#Filter the dataset with 12 airports
df2 = df.loc[df['ORIGIN_AIRPORT_ID'].isin([11298,12478,11278,10397,14747,10821,13204,12892,14771,12264,11292,13930])]


#export to csv
df2.to_csv("flight_data_4_airlines_12_departure_airports.csv", index=False, sep=',', encoding='utf-8')

