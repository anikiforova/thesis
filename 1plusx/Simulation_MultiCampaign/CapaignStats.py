import pandas as pd
import os 
import glob
import math
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv
#import matplotlib.pyplot as plt

import sys
from os import path
# to be able to access sister folders
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import Algos.Util as Util
from Algos.Metadata import Metadata 

campaign_ids = [597165, 837817, 722100] #809153

output = open("{}/CampaignStats.csv".format(Metadata.base_path), "w")
output.write("CampaignId,CTR,Impressions,UserCount\n")
for campaign_id in campaign_ids:
	print("Starting campaign: {}...".format(campaign_id))
	meta = Metadata(campaign_id)
	file_name = "{0}/sorted_time_impressions.csv".format(meta.path)
	impressions = read_csv(file_name, sep=",", header=0, dtype={"UserHash": str, "Click": np.int32, "Timestamp": np.int32})

	click_info = impressions["Click"].values
	ctr = np.mean(click_info)
	impression_count = len(click_info)
	distinct_users = impressions.UserHash.unique()
	count_distinct_users = len(distinct_users)
	output.write("{},{:.04},{},{}\n".format(campaign_id, ctr, impression_count, count_distinct_users))

output.close()