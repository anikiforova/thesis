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

# 1.2 for 0.02 
alphas 						= [1] # 0.5
user_recommendation_part 	= [0.2] 
user_train_part 			= [0.2]
eq_clicks					= [True]

meta = Metadata()

algoName = "GP_Clustered"

output_path = "./Results/{0}/{1}.csv".format(meta.campaign_id, algoName)
if not Path(output_path).is_file():
	output = open(output_path, "w")	
	output.write("Clicks,Impressions,TotalImpressions,Method,RecommendationSizePercent,Timestamp,Alpha,TrainPart,MSE,CumulativeMSE,AdditionalInfo\n")
else:
	output = open(output_path, "a")

# specials = read_csv("{0}//special_users.csv".format(meta.path), header=0)
# specials = set(specials["UserHash"].values)
# specials_count = len(specials)
output_algoName = meta.construct_algo_name(algoName)
algo = GP_Clustered(meta)

for eq in eq_clicks:
	meta.equalize_clicks = eq
	for alpha in alphas:
		for recommendation_part, train_part in zip(user_recommendation_part, user_train_part):
			testMeta = TestMetadata(meta)
			testMeta.set_local_params(alpha, recommendation_part, train_part)
			algo.setup(testMeta)

			additional_info = testMeta.get_additional_info()

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
			
			recommended_users = list()
			for total_impressions, line in enumerate(input):
				parts = line.split(",")
				user_id = int(parts[0])
				click = int(parts[1])
				timestamp_raw = int(parts[2])/1000
				timestamp = datetime.datetime.fromtimestamp(timestamp_raw)
				#print("User{0} click{1} timestamp{2}".format(user_id, click, timestamp))
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
					new_recommended_users 	= algo.get_recommendations(recommendation_part)
					# intersection = new_recommended_users.intersection(recommended_users)
					recommended_users = new_recommended_users
					print(len(recommended_users))
					# recommendation_size = float(len(recommended_users))
					# special_intersection = specials.intersection(new_recommended_users)
					# print( "Intersection with: Old:{:.3%} Specials:{:.3%}".format((float(len(intersection)) /recommendation_size  ), float(len(special_intersection))/specials_count))
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

					print('{:.2%} ImpC:{} ClkC:{} CumCTR:{:.3%} CTR:{:.3%} CumMC:{:.3%} MC:{:.3%} Overlap:{} UniqueUsers:{}'.format(total_impressions/meta.total_lines, int(impression_count), int(click_count), click_count/impression_count, local_clicks/local_count, missed_clicks/total_clicks, local_missed_clicks/total_local_clicks, int(total_local_count), unique_users_seen))

					output.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n".format(click_count, impression_count, total_impressions, output_algoName, recommendation_part, timestamp_raw, alpha, train_part,local_SE/10000, cumulative_SE/total_impressions, additional_info))
					output.flush()

					local_SE = 0.0	
					local_clicks = 0.0	
					local_count = 1.0
					local_missed_clicks = 0.0
					total_local_clicks = 0.0
					SE = 0.0
					
			input.close()
				
output.close()


















