import pandas as pd
import os 
import glob
import numpy as np
from pandas import read_csv
from pandas import DataFrame
import pyarrow.parquet as pq

range_of_dates = pd.date_range(start='15/8/2018', end='28/08/2018')

all_impressions = pd.DataFrame(columns=list(["CampaignId", "UserHash", "Click", "Timestamp"]))

for date in range_of_dates:
	date = date.strftime("%Y-%m-%d")
	print(date)
	file_name = "../../RawData/Campaigns/5/Processed/sorted_time_impressions_{0}.csv".format(date)
	
	data = read_csv(file_name, ",")
	all_impressions = all_impressions.append(data, ignore_index=True)
	
all_impressions = all_impressions.sort_values(by=['Timestamp'])
all_impressions.to_csv("../../RawData/Campaigns/5/Processed/all_impressions.csv", mode="w", sep=",", header=True, index=False, \
		columns=["CampaignId", "UserHash", "Click", "Timestamp"])

# there are aparently NAN values in timestamp for some campaigns
