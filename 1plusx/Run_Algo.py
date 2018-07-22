import numpy as np
import random 
import math
import time
import datetime
from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt

from Metadata import Metadata

from Random import Random
from Regression import Regression
from LinUCB_Disjoint import LinUCB_Disjoint
from TS_Lin import TS_Lin
from GP_Clustered import GP_Clustered
from NN import NN

# 1.2 for 0.02 
alphas 						= [0.001, 0.0001, 0.01]
user_recommendation_part 	= [0.02, 0.05] 
user_train_part 			= [0.02, 0.05]

print_output 				= True
print_mse 					= False

meta = Metadata()

algoName = "TS_Lin"

if print_output:
	output = open("./Results/{0}.csv".format(algoName), "a")
	# output.write("Clicks,Impressions,TotalImpressions,Method,RecommendationSizePercent,Timestamp,Alpha,TrainPart,MSE,CumulativeMSE\n")

if print_mse:
	output_mse = open("./Results/MSE/{0}.csv".format(algoName), "a")
	# output_mse.write("MSE,CumulativeMSE,TotalImpressions,Method,RecommendationSizePercent,Timestamp,Alpha,TrainPart\n")

output_algoName = meta.construct_algo_name(algoName)

specials = read_csv("{0}//special_users.csv".format(meta.path), header=0)
specials = set(specials["UserHash"].values)
specials_count = len(specials)

algo = TS_Lin(meta)

for alpha in alphas:
	for recommendation_part, train_part in zip(user_recommendation_part, user_train_part):
		meta.set_local_params(alpha, recommendation_part, train_part)

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
	
		print("Starting evaluation of {} with recommendation of size: {:.2%} train size {:.2%} and alpha: {}".format(algoName, recommendation_part, train_part, alpha))
		input = open("{0}//sorted_time_impressions.csv".format(meta.path), "r")
		input.readline() # get rid of header
		line = input.readline()
		hour_begin_timestamp = datetime.datetime.fromtimestamp(int(line.split(",")[2])/1000)
		
		algo.setup()

		recommended_users = list()
		for total_impressions, line in enumerate(input):
			parts = line.split(",")
			user_id = int(parts[0])
			click = int(parts[1])
			timestamp_raw = int(parts[2])/1000
			timestamp = datetime.datetime.fromtimestamp(timestamp_raw)
			
			impressions_per_recommendation_group += 1
			cur_SE 			= (click - algo.getPrediction(user_id)) ** 2
			SE 				+= cur_SE 
			local_SE 		+= cur_SE
			cumulative_SE 	+= cur_SE

			if warmup and (timestamp - hour_begin_timestamp).seconds < meta.time_between_updates_in_seconds: 
				users_to_update.append(user_id)
				clicks_to_update.append(click)
				SE = 0.0
				cumulative_SE = 0.0
				continue	

			if (timestamp - hour_begin_timestamp).seconds >= meta.time_between_updates_in_seconds and len(clicks_to_update) > 1000:
				# 	if print_mse:
				# 		MSE = SE / impressions_per_recommendation_group
				# 		cumulative_MSE = cumulative_SE / total_impressions
				# 		# output_mse.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(MSE, cumulative_MSE, total_impressions, algoName, part, timestamp_raw, alpha, train_part))
				# 		output_mse.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(MSE, cumulative_MSE, total_impressions, output_algoName, recommendation_part, timestamp_raw, alpha))
				# 		output_mse.flush()
				# 	SE = 0.0
				# 	impressions_per_recommendation_group = 0.0
				# 	# update based on train part
				
				algo.update(users_to_update, clicks_to_update)
				new_recommended_users 	= algo.get_recommendations(recommendation_part)
				intersection = new_recommended_users.intersection(recommended_users)
				recommended_users = new_recommended_users
				recommendation_size = float(len(recommended_users))
				special_intersection = specials.intersection(new_recommended_users)
				print( "Intersection with: Old:{:.3%} Specials:{:.3%}".format((float(len(intersection)) /recommendation_size  ), float(len(special_intersection))/specials_count))
				train_users = recommended_users
				
				if recommendation_part != train_part:
					train_users 		= algo.get_recommendations(train_part)
				
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

				print('{:.2%} ImpC:{} ClkC:{} CumCTR:{:.3%} CTR:{:.3%} CumMC:{:.3%} MC:{:.3%} Overlap:{} UniqueUsers:{}'.format(total_impressions/meta.total_lines, int(impression_count), int(click_count), click_count/impression_count, local_clicks/local_count, missed_clicks/total_clicks, local_missed_clicks/total_local_clicks, total_local_count, unique_users_seen))

				if print_output:
					output.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}\n".format(click_count, impression_count, total_impressions, output_algoName, recommendation_part, timestamp_raw, alpha, train_part,local_SE/10000, cumulative_SE/total_impressions))
					output.flush()

				local_SE = 0.0	
				local_clicks = 0.0	
				local_count = 1.0
				local_missed_clicks = 0.0
				total_local_clicks = 0.0
				SE = 0.0
				
		input.close()
				
if print_output:
	output.close()


















