import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime
from pandas import read_csv

campaignIds = set([866128, 856805, 847460, 858140, 865041])

a = pd.date_range(start='15/8/2018', end='28/08/2018')

add_global_header = True
for date in a:
	date = date.strftime("%Y-%m-%d")
	print("Starting Date: {}.".format(date))
	input_impression_file_name = "../../RawData/Campaigns/10/Processed/sorted_time_impressions_{0}.csv".format(date)
	output_impression_file_name = "../../RawData/Campaigns/5/Processed/sorted_time_impressions_{0}.csv".format(date)
	
	input_users_file_name = "../../RawData/Campaigns/10/Processed/all_users_{0}.csv".format(date)
	output_users_file_name = "../../RawData/Campaigns/5/Processed/all_users_{0}.csv".format(date)
	
	print("\tReading impressions...")	
	impressions = read_csv(input_impression_file_name, header=0)

	print("\tFiltering impressions...")
	impressions = impressions[impressions['CampaignId'].isin(campaignIds)]
	impressions = impressions.sort_values(by=['Timestamp'])

	print("\tOutputing impressions...")
	impressions.to_csv(output_impression_file_name, mode="w", sep=",", header=True, index=False, \
		columns=["CampaignId", "UserHash", "Click", "Timestamp"])

	unique_user_ids = np.unique(np.array(impressions["UserHash"]))

	print("\tReading users ...")
	users = read_csv(input_users_file_name, header=0)
	print("\tFiltering users ...")
	users = users[users["UserHash"].isin(unique_user_ids)]

	print("\tOutputting users ...")
	users.to_csv(output_users_file_name, mode="w", sep=",", header=True, index=False, \
		columns=["UserEmbedding","UserHash"])
