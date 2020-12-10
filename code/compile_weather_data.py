import pandas as pd  # for data frame creation
import os  # for OS interface (to get/change directory)
import sys

path = input ("Path to CSV files: ")

try:
    os.chdir(path)
except:
    print("Unable to find path specified. Reverting to default path")
    try:
        os.chdir('C:/Users/jmmen/Documents/datasets')
    except:
        sys.exit("Unable to reach default path. Aborting")
print("Current Path: " + os.getcwd())

weatherdata = None
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for file in files:
    if file.endswith(".csv") and not file.endswith("Complete_Raw.csv") and not file.endswith("PercentPopulated.csv"):
        print("Attempting to load CSV file: " + file)
        if weatherdata is None:
            weatherdata = pd.read_csv(file, sep=",", dtype=('str'))
            print("Imported CSV file with {} columns and {} rows".format(weatherdata.shape[1], weatherdata.shape[0]))
        else:
            mydata = pd.read_csv(file, sep=",", dtype=('str'))
            print("Imported CSV file with {} columns and {} rows".format(mydata.shape[1], mydata.shape[0]))
            print("Merging dataframes")
            weatherdata = pd.concat([weatherdata, mydata], ignore_index=True, sort=True)
            print("Merge successful")
            
print("Final dataframe dimensions: {} columns and {} rows".format(weatherdata.shape[1], weatherdata.shape[0]))
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

print("\nExporting results to 'Weather_Data_Complete_Raw.csv'")
print("Done \n")
weatherdata.to_csv("Weather_Data_Complete_Raw.csv", sep=',', encoding='utf-8', index=False)

percentpopulated = .70
print("Filtering out columns where less than {}% of rows are populated".format(str(percentpopulated*100)))
percentnull = weatherdata.isnull().sum()/len(weatherdata)
lownulls = percentnull[percentnull <= (1-percentpopulated)]

print("Total columns: {}".format(weatherdata.shape[1]))
print("Columns with at least {}% of rows populated: {}".format(percentpopulated*100, len(lownulls)))

filteredweatherdata = weatherdata[lownulls.keys()]
print("First 3 rows:")
print(filteredweatherdata.head(3))

print("\nExporting results to 'Weather_Data_Complete_Raw_{}PercentPopulated.csv'".format(str(percentpopulated*100).split(".")[0]))
filteredweatherdata.to_csv("Weather_Data_Complete_Raw_{}PercentPopulated.csv".format(str(percentpopulated*100).split(".")[0]), sep=',', encoding='utf-8', index=False)
print("Done")