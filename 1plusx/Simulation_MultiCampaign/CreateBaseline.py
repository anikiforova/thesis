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


campaign_ids = [597165, 837817, 722100] # 809153
campaign_stats = read_csv("{}/CampaignStats.csv".format(Metadata.base_path), sep=",", header=0, index_col=0,
	dtype={"CampaignId": np.int32, "CTR": np.float32, "Impressions": np.int32, "UserCount": np.int32})

#x.iloc[1] = dict(x=9, y=99)
meta = Metadata(0)
input = open("{0}/sorted_time_impressions.csv".format(meta.path), "r")
input.readline()

output = open("{0}/multi_base_sorted_time_impressions.csv".format(meta.path), "w")
output.write("UserHash,Timestamp,{}\n".format(",".join(str(c) for c in campaign_ids)))

for line_index, line in enumerate(input):
	line_parts = line.split(",")
	oritinal_campaign_id = int(line_parts[3])

	output_line = "{0},{1}".format(line_parts[0], line_parts[2])
	for index, campaign_id in enumerate(campaign_ids):
		impression = line_parts[1]
		if campaign_id != oritinal_campaign_id:
			impression = 0 if campaign_stats.iloc[index]["CTR"] < np.random.uniform(0,1) else 1

		output_line = "{},{}".format(output_line, impression)

	output.write("{}\n".format(output_line))
	if line_index % 10000 == 0:
		print(".", end='', flush=True)	
		output.flush()

output.close()