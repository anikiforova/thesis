import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv

import sys
from os import path
# sys.path.append('../../plusx')
print(path.dirname( path.dirname( path.abspath(__file__) ) ))
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import Algos.Util as Util
from Algos.Metadata import Metadata 
from Algos.Regression import Regression 
from Algos.TestMetadata import TestMetadata 

campaign_ids = [597165, 722100, 837817]#, 809153]
folder_name = 0
algos = list()

path = "../../RawData/Campaigns/"
impressions_file_path_extension = "/Processed/sorted_time_impressions.csv"
user_file_path_extension = "/Processed/all_users.csv"

for campaign_id in campaign_ids:
	print ("Starting model for {}..".format(campaign_id))
	impressions_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)
	user_file_path 		  = "{0}/{1}/{2}".format(path, campaign_id, user_file_path_extension)

	impressions = read_csv(impressions_file_path, ",")
	
	users = impressions["UserHash"].values
	clicks = impressions["Click"].values

	meta = Metadata(campaign_id)
	algo = Regression(meta)
	testMeta = TestMetadata(meta)
	testMeta.click_percent = 0.0
	algo.setup(testMeta)
	algo.fit(users, clicks)
	algos.append(algo)
	print("Done")
	
meta = Metadata(folder_name)
user_ids, user_embeddings = meta.read_user_embeddings()
user_hash_to_index = dict(zip(user_ids, range(0, len(user_ids))))

output = open("{0}//regression_sorted_time_impressions.csv".format(meta.path), "w")
input = open("{0}//sorted_time_impressions.csv".format(meta.path), "r")
print(meta.path)
header = input.readline() # get rid of header

str_campaign_ids = ','.join(str(c) for c in campaign_ids)
output.write(header.rstrip() + "," + str_campaign_ids + "\n")
print("Starting output...", end='', flush=True)

predictions = dict()
for campaign_id, algo in zip(campaign_ids, algos):
	print("Starting prediction for {0}".format(campaign_id))
	prediction = algo.model.predict(user_embeddings)
	predictions[campaign_id] = prediction

for index, line in enumerate(input):
	user_hash,_,_,_ = Util.get_line_info(line) 
	user_index = user_hash_to_index[user_hash]
	output_line = line.rstrip()

	for campaign_id in campaign_ids:
		output_line = "{},{:.03}".format(output_line, predictions[campaign_id][user_index])

	output.write(output_line + "\n")
	if index % 1000 == 0:
		output.flush()
		print(".", end="", flush=True)

print("Done")
output.close()
input.close()






