import pandas as pd
import os 
import glob
import numpy as np
from pandas import read_csv
from pandas import DataFrame
import pyarrow.parquet as pq


all_impressions = read_csv("../../RawData/Campaigns/5/Processed/all_impressions.csv")
campaig_ids = np.unique(np.array(all_impressions["CampaignId"].values))
print(campaig_ids)
for campagn_id in campaig_ids:
	print("Starting with campagn_id: {}".format(campagn_id))
	cur_impressions = all_impressions[all_impressions["CampaignId"] == campagn_id]
	cur_impressions.to_csv("../../RawData/Campaigns/{}/Processed/sorted_time_impressions.csv".format(campagn_id), mode="w", sep=",", \
		header=True, index=False, columns=["UserHash", "Click", "Timestamp"])
