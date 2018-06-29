import pandas as pd
import os 
import glob
import numpy as np
from pandas import read_csv 

file_name = "../../RawData/Campaigns/809153/Processed/time_impressions.csv"
file_name_sorted = "../../RawData/Campaigns/809153/Processed/sorted_time_impressions.csv"
data = read_csv(file_name, ",",dtype={"Timestamp": np.int64, "UserHash":str, "Click":str}, header='infer')

result = data.sort_values("Timestamp")

# print(result.head(10))
result.to_csv(file_name_sorted, ",", index=False, mode='w')
