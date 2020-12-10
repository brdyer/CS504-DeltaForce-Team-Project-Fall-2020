import numpy as np
import pandas as pd  # for data frame creation
import multiprocessing as mp  # for parallel processing
import os  # for OS interface (to get/change directory)
import sys

### Define functions
### -------------------------------------------------- ###

def convert_to_float_nan(val):
    if pd.isna(val) or pd.isnull(val) or val == "":
        return np.nan
    try:
        return float(val);
    except ValueError:
        print("Invalid float ", val, "; returning NaN")
        return np.nan;

### -------------------------------------------------- ###

def convert_to_float_zero(val):
    if pd.isna(val) or pd.isnull(val) or val == "":
        return float(0.00)
    try:
        return float(val);
    except ValueError:
        print("Invalid float ", val, "; returning 0.00")
        return float(0.00);

### -------------------------------------------------- ###

def calculate_average_degrees(val1, val2):
    if abs(val1 - val2) > 180:
        avg = ((val1 + val2 + 360) / 2)
        if (avg > 360):
            return avg - 360
        else:
            return avg
    else:
        return ((val1 + val2) / 2)

### -------------------------------------------------- ###

def update_missing(df, column_name):
    df_id = df["STATION"].iloc[0]
    length = df.shape[0]    
    # Try to retrieve first value, if error, set it to 0.00
    prior_val = convert_to_float_zero(df[column_name].iloc[0])
    current_val = convert_to_float_nan(df[column_name].iloc[1])
    df.at[0, column_name] = prior_val
    
    for x in range(1, length - 1):
        print("Processing station ", df_id, " index ", x, " of column ", column_name)
        next_val = convert_to_float_nan(df[column_name].iloc[x + 1])
        if pd.isna(current_val):
            if not pd.isna(next_val):
                current_val = ((prior_val + next_val)/2)
            else:
                current_val = prior_val
        df.at[x, column_name] = current_val
        prior_val = current_val
        current_val = next_val
    if not pd.isna(current_val):
        df.at[length - 1, column_name] = current_val
    else:
        df.at[length - 1, column_name] = prior_val
    df[column_name] = pd.to_numeric(df[column_name])
    return df;

### -------------------------------------------------- ###

def update_missing_degree(df, column_name):
    df_id = df["STATION"].iloc[0]
    length = df.shape[0]    
    # Try to retrieve first value, if error, set it to 0.00
    prior_val = convert_to_float_zero(df[column_name].iloc[0])
    current_val = convert_to_float_nan(df[column_name].iloc[1])
    df.at[0, column_name] = prior_val
    
    for x in range(1, length - 1):
        print("Processing station ", df_id, " index ", x, " of column ", column_name)
        next_val = convert_to_float_nan(df[column_name].iloc[x + 1])
        if pd.isna(current_val):
            if not pd.isna(next_val):
                current_val = calculate_average_degrees(prior_val, next_val)
            else:
                current_val = prior_val
        df.at[x, column_name] = current_val
        prior_val = current_val
        current_val = next_val
    if not pd.isna(current_val):
        df.at[length - 1, column_name] = current_val
    else:
        df.at[length - 1, column_name] = prior_val
    df[column_name] = pd.to_numeric(df[column_name])
    return df;

### -------------------------------------------------- ###

def process_station(station):
    fields_to_process = ["HourlyAltimeterSetting","HourlyDewPointTemperature",
                         "HourlyDryBulbTemperature","HourlyPrecipitation",
                         "HourlyRelativeHumidity","HourlySeaLevelPressure",
                         "HourlyStationPressure","HourlyVisibility",
                         "HourlyWetBulbTemperature","HourlyWindSpeed"]
    fields_to_process_degree = ["HourlyWindDirection"]
    
    station.reset_index(drop=True, inplace=True)
    # Convert DATE to a date format and sort dataset by DATE
    station['DATE'] = pd.to_datetime(station['DATE'])
    station = station.sort_values('DATE', ascending=True)

    for field in fields_to_process:
        # Replace trace amount (T) with 0.00
        station[field] = station[field].replace('T', 0.00)
        # Remove s from data
        station[field] = station[field].str.replace('s', '')
        # Process data for this column
        station = update_missing(station, field)

    for field in fields_to_process_degree:
        # Replace trace amount (T) with 0.00
        station[field] = station[field].replace('T', 0.00)
        # Remove s from data
        station[field] = station[field].str.replace('s', '')
        # Process data for this column
        station = update_missing_degree(station, field)
    return station;

### -------------------------------------------------- ###

def applyParallel(dfGrouped, func):
    dflist = [group for _, group in dfGrouped]
    print("Groups: ", len(dflist))
    pool = mp.Pool(mp.cpu_count())
    ret_list = pool.map(func, dflist)
    return pd.concat(ret_list, ignore_index=True)

### -------------------------------------------------- ###
### Begin MAIN

def main():
    path = input("Path to CSV file: ")
    try:
        os.chdir(path)
    except:
        print("Unable to find path specified. Reverting to default path")
        try:
            os.chdir('C:/Users/jmmen/Documents/datasets')
        except:
            sys.exit("Unable to reach default path. Aborting")
    print("Current Path: " + os.getcwd())
    
    wd = None
    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for file in files:
        if file.startswith("Weather_Data_Complete_Raw_") and file.endswith("PercentPopulated.csv"):
            try:
                print("\nReading file: ", file)
                wd = pd.read_csv(file, sep=",", dtype=('str'))
                break
            except:
                sys.exit("Unable to read file. Aborting.")
    else:
        if wd == None:
            sys.exit("No appropriate file found in this directory. Please ensure that the output of the 'compile_flight_data.py' script is in this directory")
    print("Dataframe dimensions: {} columns and {} rows".format(wd.shape[1], wd.shape[0]))
    
    # Replace all blanks with NaN, Remove any rows where DATE or STATION is NaN
    print("Replacing blank records with NaN")
    wd = wd.replace(r'^\s*$', np.nan, regex=True)
    wd = wd[wd['DATE'].notna()]
    wd = wd[wd['STATION'].notna()]
    
    # Group df by stations and process in parallel
    print("Beginning processing")
    wd = applyParallel(wd.groupby(["STATION"]), process_station)

    wd.to_csv("Weather_Data_Cleaned.csv", sep=',', encoding='utf-8', index=False)

### End MAIN
### -------------------------------------------------- ###

if __name__ == "__main__":
    main()
