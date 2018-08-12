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

from Random import Random
from Regression import Regression
from LinUCB_Disjoint import LinUCB_Disjoint
from TS_Lin import TS_Lin
from GP_Clustered import GP_Clustered
from NN import NN

import TestBuilder
import Util
import SimulationMetrics

campaign_id = 837817
meta = Metadata(campaign_id)

algoName = "TS_Lin"
algo = TS_Lin(meta)

testsMeta = TestBuilder.get_lin_tests_mini(meta)

for simulation_id in np.arange(0, 1):
	simulation_id = 4
	output_path = "./Results/{0}/Simulated/{1}/{2}test.csv".format(meta.campaign_id, simulation_id, algoName)
	output_column_names = False
	if not Path(output_path).is_file():
		output = open(output_path, "w")	
		output_column_names = True;
	else:
		output = open(output_path, "a")

	impressions_file_path = "{0}//simulated_time_impressions_s{1}.csv".format(meta.path, simulation_id)
	impressions = read_csv(impressions_file_path, ",")
	clicks = impressions["Click"].values
	background_ctr = np.mean(clicks)

	for testMeta in testsMeta:
		algo.setup(testMeta)
		if output_column_names:
			output.write("Clicks,Impressions,TotalImpressions,Method,Timestamp,MSE,CumulativeMSE,MMSE,MCumulativeMSE,LocalCTR,Calibration,NE,RIG,{}\n".format(testMeta.get_additional_column_names()))
			output_column_names = False

		impression_count 	= 1.0
		click_count 		= 0.0
		local_clicks 		= 0.0
		local_count 		= 1.0
		total_local_count 	= 1.0
		missed_clicks 		= 0.0
		total_clicks 		= 1.0
		local_missed_clicks = 0.0
		total_local_clicks 	= 1.0
		impressions_per_recommendation_group = 0.0
		
		SE = 0.0
		local_modified_SE = 0.0
		local_SE = 0.0
		cumulative_SE = 0.0 
		modified_cumulative_SE = 0.0

		warmup = True

		users_to_update = list()
		clicks_to_update = list()
		all_common_impressions = list()
		predicted_values = list()
		recommended_users = list()

		print("Starting evaluation of {} with {}".format(algoName, testMeta.get_additional_info()))

		input = open(impressions_file_path, "r")
		input.readline() # get rid of header
		_, _, _, hour_begin_timestamp = Util.get_line_info(input.readline())
		
		for total_impressions, line in enumerate(input):
			user_id, click, timestamp_raw, timestamp = Util.get_line_info(line)

			impressions_per_recommendation_group += 1
			prediction = algo.getPrediction(user_id)
			modified_click_value = click * 300

			cur_SE 					= (click - prediction) ** 2
			cur_modified_SE 		= (modified_click_value - prediction) ** 2

			local_SE 				+= cur_SE
			cumulative_SE 			+= cur_SE

			local_modified_SE 		+= cur_modified_SE
			modified_cumulative_SE 	+= cur_modified_SE

			if warmup and (timestamp - hour_begin_timestamp).seconds < testMeta.time_between_updates_in_seconds: 
				users_to_update.append(user_id)
				clicks_to_update.append(click)
				cumulative_SE = 0.0
				modified_SE = 0.0
				modified_cumulative_SE = 0.0
				continue	

			if (timestamp - hour_begin_timestamp).seconds >= testMeta.time_between_updates_in_seconds:			
				algo.update(users_to_update, clicks_to_update)
				recommended_users = algo.get_recommendations(testMeta.recommendation_part)
					
				users_to_update = list()
				clicks_to_update = list()
				hour_begin_timestamp = timestamp
				total_local_count = 1.0
				warmup = False
				continue

			if user_id in recommended_users:	
				impression_count += 1
				local_count += 1
				click_count += click
				local_clicks += click
				users_to_update.append(user_id)
				clicks_to_update.append(click)	
				predicted_values.append(prediction)
				all_common_impressions.append(click)
			else:
				missed_clicks += click
				local_missed_clicks += click

			total_clicks += click
			total_local_clicks += click

			# print('.', end='', flush=True)	
			print_impressions = 10000
			if total_impressions % print_impressions == 0 and len(users_to_update) > 0:
				unique_users_seen = len(set(users_to_update))
				total_local_count += local_count

				print('{:.2%} ImpC:{} ClkC:{} CumCTR:{:.3%} CTR:{:.3%} CumMC:{:.3%} MC:{:.3%} Overlap:{} UniqueUsers:{}'.format(total_impressions/meta.total_lines[campaign_id], int(impression_count), int(click_count), click_count/impression_count, local_clicks/local_count, missed_clicks/total_clicks, local_missed_clicks/total_local_clicks, int(total_local_count), unique_users_seen))

				cur_ctr, Calibration, NE, RIG = SimulationMetrics.get_simulation_metrics(all_common_impressions, algo.prediction, background_ctr)

				output.write("{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}\n".format(
					click_count, 
					impression_count, 
					total_impressions,
					algoName, 
					timestamp_raw, 
					local_SE / print_impressions, 
					local_modified_SE / print_impressions, 
					cumulative_SE / total_impressions, 
					modified_cumulative_SE / total_impressions,
					cur_ctr,
					Calibration,
					NE,
					RIG, 
					testMeta.get_additional_column_info()))
				output.flush()

				local_SE = 0.0	
				local_clicks = 0.0	
				local_count = 1.0
				local_missed_clicks = 0.0
				local_modified_SE = 0.0
				total_local_clicks = 1.0
				
		input.close()
					
	output.close()


















