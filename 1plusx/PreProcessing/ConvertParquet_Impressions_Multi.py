import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime

a = pd.date_range(start='15/8/2018', end='28/08/2018')

add_global_header = True
for date in a:
	date = date.strftime("%Y-%m-%d")

	path = "../../RawData/Campaigns/10/Impressions/{0}/"
	
	full_path = path.format(date)
	file_path = full_path + "/{0}"

	all_files = glob.glob(full_path + "*.parquet")
	all_files.sort()

	add_header = True
	for file_name in all_files:
		file_index = file_name.split("/")[-1].split(".")[0]

		print("Reading file {0}".format(file_name))
		file = pd.read_parquet(file_name, engine='pyarrow')
		data = file.rename(index=str, columns={"campaignId":"CampaignId", "hash":"UserHash", "click":"Click", "timestamp":"Timestamp"})

		#data['Timestamp'] = data['event'].apply(lambda x: x.get('timestamp'))
		data["Click"] = data['Click'].apply(lambda x: ((x == "Click") * 1) + (x == "Conversion") * 2)
		
		data.to_csv( "../../RawData/Campaigns/10/Processed/Impressions_{0}.csv".format(date), mode="a", sep=",", header=add_header, \
			index=False, columns=["CampaignId", "UserHash", "Click", "Timestamp"])

		# data.to_csv( "../../RawData/Multi-Campaign/Processed/all_impressions.csv", mode="a", sep=",", header=add_global_header, \
		# 	index=False, columns=["CampaignId", "UserHash", "Click", "Timestamp"])


		add_header = False
		add_global_header = False
