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
import Algos.MetricsCalculator as Metrics
from Algos.Metadata import Metadata 

lower_multi_campaign_path = "../../RawData/Multi-Campaign/Processed/SimulationLower"
real_multi_campaign_path = "../../RawData/Multi-Campaign/Processed/FiveCampaigns"

campaign_ids = [866128, 856805, 847460, 858140, 865041] # 809153
campaign_stats = read_csv("{}/CampaignStats.csv".format(Metadata.base_path), sep=",", header=0, index_col=0,
	dtype={"CampaignId": np.int32, "CTR": np.float32, "Impressions": np.int32, "UserCount": np.int32})

a = pd.date_range(start='15/8/2018', end='28/08/2018')

printHeader = True
# print(campaign_stats)

campaign_ctr = campaign_stats["CTR"].values
# print(campaign_ctr)
for date in a:
	date = date.strftime("%Y-%m-%d")
	print("\nDate {}...".format(date))

	input = open("{0}/sorted_time_impressions_{1}.csv".format(real_multi_campaign_path, date), "r")
	input.readline()

	output = open("{0}/sorted_time_impressions_{1}.csv".format(lower_multi_campaign_path, date), "w")
	output.write("UserHash,Timestamp,{}\n".format(",".join(str(c) for c in campaign_ids)))

	random_values = np.random.uniform(0, 1, 10000 * len(campaign_ids)).reshape([10000, len(campaign_ids)])
	for line_index, line in enumerate(input):
		line_parts = line.split(",")
		oritinal_campaign_id = int(line_parts[0])
		user_hash = int(line_parts[1])
		click = line_parts[2]
		timestamp = line_parts[3][0:-1]

		output_line = "{0},{1}".format(user_hash, timestamp)
		results = list()
		for campaign_index, campaign_id in enumerate(campaign_ids):
			if campaign_id != oritinal_campaign_id:
				click = "0" if campaign_ctr[campaign_index] < random_values[line_index%10000][campaign_index] else "1"
			results.append(click)
		
		output.write("{0},{1},{2}\n".format(user_hash, timestamp, ",".join(results)))
		if line_index % 10000 == 0:
			random_values = np.random.uniform(0, 1, 10000 * len(campaign_ids)).reshape([10000, len(campaign_ids)])
			print(".", end='', flush=True)	
			output.flush()

	output.close()