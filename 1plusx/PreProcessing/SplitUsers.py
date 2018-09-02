import pandas as pd
import os 
import glob
import numpy as np
from pandas import read_csv
from pandas import DataFrame
import pyarrow.parquet as pq


all_impressions = read_csv("../../RawData/Multi-Campaign/Processed/5_all_impressions.csv")
campaig_ids = np.unique(np.array(all_impressions["CampaignId"].values))
filter_date_range = pd.date_range(start='15/8/2018', end='28/08/2018')

print(campaig_ids)
for campagn_id in campaig_ids:
	print("Starting with campagn_id: {}".format(campagn_id))
	cur_impressions = all_impressions[all_impressions["CampaignId"] == campagn_id]
	cur_user_hash = set(cur_impressions["UserHash"].values)

	unique_user_ids = np.unique(np.array(cur_impressions["UserHash"]))

	add_global_header = True
	all_users = pd.DataFrame(columns=list(["UserEmbedding", "UserHash"]))
	for date in filter_date_range:
		date = date.strftime("%Y-%m-%d")
		print("Reading user file: {}".format(date))

		users = read_csv("../../RawData/Multi-Campaign/Processed/5_Users_{}.csv".format(date), ",")
		users = users[users["UserHash"].isin(unique_user_ids)]
		all_users = all_users.append(users, ignore_index=True)
	
	print("Outputting user file for campaign: {}".format(campagn_id))
	all_users = all_users.groupby(['UserHash'])['UserEmbedding', 'UserHash'].head(1)
	all_users.to_csv("../../RawData/Campaigns/{}/Processed/all_users.csv".format(campagn_id), mode="w", sep=",", \
		header=True, index=False, columns=["UserEmbedding", "UserHash"])
