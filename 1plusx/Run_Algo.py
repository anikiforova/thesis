import numpy as np
import random 
import math
import time
import datetime
from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt
from pathlib import Path

from Metadata import Metadata
from TestMetadata import TestMetadata
import TestBuilder
import Util

from Random import Random
from Regression import Regression
from LinUCB_Disjoint import LinUCB_Disjoint
from TS_Lin import TS_Lin
from GP_Clustered import GP_Clustered
from NN import NN

meta = Metadata()
algoName = "LinUCB_Disjoint"

testsMeta = TestBuilder.get_lin_tests(meta)
algo = LinUCB_Disjoint(meta)

output_path = "./Results/{0}/{1}_New.csv".format(meta.campaign_id, algoName)
output_column_names = False
if not Path(output_path).is_file():
	output = open(output_path, "w")	
	output_column_names = True;
else:
	output = open(output_path, "a")

# specials = read_csv("{0}//special_users.csv".format(meta.path), header=0)
# specials = set(specials["UserHash"].values)
# specials_count = len(specials)

for testMeta in testsMeta:
	algo.setup(testMeta)
	if output_column_names:
		output.write("Clicks,Impressions,TotalImpressions,Method,Timestamp,MSE,CumulativeMSE,{}\n".format(testMeta.get_additional_column_names()))
		output_column_names = False

	impression_count 	= 1.0
	click_count 		= 0.0
	local_clicks 		= 0.0
	local_count 		= 1.0
	total_local_count 	= 0.0
	missed_clicks 		= 0.0
	total_clicks 		= 0.0
	local_missed_clicks = 0.0
	total_local_clicks 	= 0.0
	impressions_per_recommendation_group = 0.0
	
	SE = 0.0
	local_SE = 0.0
	cumulative_SE = 0.0 

	warmup = True

	users_to_update = list()
	clicks_to_update = list()
	recommended_users = list()

	print("Starting evaluation of {} with {}".format(algoName, testMeta.get_additional_info()))

	input = open("{0}//sorted_time_impressions.csv".format(meta.path), "r")
	input.readline() # get rid of header
	_, _, _, hour_begin_timestamp = Util.get_line_info(input.readline())
	
	for total_impressions, line in enumerate(input):
		user_id, click, timestamp_raw, timestamp = Util.get_line_info(line)

		impressions_per_recommendation_group += 1
		cur_SE 			= (click - algo.getPrediction(user_id)) ** 2
		SE 				+= cur_SE 
		local_SE 		+= cur_SE
		cumulative_SE 	+= cur_SE

		if warmup and (timestamp - hour_begin_timestamp).seconds < testMeta.time_between_updates_in_seconds: 
			users_to_update.append(user_id)
			clicks_to_update.append(click)
			SE = 0.0
			cumulative_SE = 0.0
			continue	

		if (timestamp - hour_begin_timestamp).seconds >= testMeta.time_between_updates_in_seconds:			
			algo.update(users_to_update, clicks_to_update)
			recommended_users = algo.get_recommendations(testMeta.recommendation_part)
			train_users = recommended_users

			if testMeta.recommendation_part != testMeta.train_part:
				train_users = algo.get_recommendations(testMeta.train_part)
			
			users_to_update = list()
			clicks_to_update = list()
			hour_begin_timestamp = timestamp
			total_local_count = 0.0
			warmup = False
			continue

		if user_id in recommended_users:	
			impression_count += 1
			local_count += 1
			click_count += click
			local_clicks += click
			if user_id in train_users:
				users_to_update.append(user_id)
				clicks_to_update.append(click)	
		else:
			missed_clicks += click
			local_missed_clicks += click

		total_clicks += click
		total_local_clicks += click

		# print('.', end='', flush=True)	
		if total_impressions % 10000 == 0:
			unique_users_seen = len(set(users_to_update))
			total_local_count += local_count

			print('{:.2%} ImpC:{} ClkC:{} CumCTR:{:.3%} CTR:{:.3%} CumMC:{:.3%} MC:{:.3%} Overlap:{} UniqueUsers:{}'.format(total_impressions/meta.total_lines, int(impression_count), int(click_count), click_count/impression_count, local_clicks/local_count, missed_clicks/total_clicks, local_missed_clicks/total_local_clicks, int(total_local_count), unique_users_seen))

			output.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(click_count, impression_count, total_impressions, algoName, timestamp_raw, local_SE/10000, cumulative_SE/total_impressions, testMeta.get_additional_column_info()))
			output.flush()

			local_SE = 0.0	
			local_clicks = 0.0	
			local_count = 1.0
			local_missed_clicks = 0.0
			total_local_clicks = 0.0
			SE = 0.0
			
	input.close()
				
output.close()


















