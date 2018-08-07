import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv


campaign_ids = [722100, 597165, 837817, 809153]
folder_name = 1

path = "../../RawData/Campaigns/"
impressions_file_path_extension = "/Processed/sorted_time_impressions.csv"

all_impressions = pd.DataFrame(columns=list(["UserHash", "Click", "Timestamp","CampaignId"]))

for campaign_id in campaign_ids:
	print("Combining campaign {0}".format(campaign_id))
	impressions_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)
	cur_impressions = read_csv(impressions_file_path, ",")
	cur_impressions["CampaignId"] = campaign_id

	all_impressions = all_impressions.append(cur_impressions, ignore_index=True)

print("Sorting impressions.")	
all_impressions = all_impressions.sort_values("Timestamp")

all_impressions.to_csv("{0}/{1}/{2}".format(path, folder_name, impressions_file_path_extension), ",", index=False, mode='w')
