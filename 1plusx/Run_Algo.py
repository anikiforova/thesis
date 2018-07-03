import numpy as np
import random 
import math
import time
import datetime
from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt

from Regression import Regression
from LinUCB_Disjoint import LinUCB_Disjoint
# from Random import Random

def normalize_dimension(dimension):
	min_value = np.min(dimension)
	max_value = np.max(dimension)
	return (dimension - min_value)/(max_value - min_value)

total_lines = 14392074
repetitions = 10

impressions_between_update = 100

alpha = 0.1
algoName = "Regression"
user_recommendation_part = [0.1]
time_between_updates_in_seconds = 60 * 60 * 1 # 1 hour

output = open("./Results/{0}_1h_no_impressions_past_click.csv".format(algoName), "a")
# output.write("Clicks,Impressions,TotalImpressions,Method,RecommendationSizePercent,RecommendationSize,Timestamp\n")

name = "809153"
path = "..//..//RawData//Campaigns"

print("Reading users.. ", end='', flush=True)	
users = read_csv("{0}//{1}//Processed//all_users.csv".format(path, name), header=0)#, index_col=1)
print(" Done.")

print("Parsing users.. ", end='', flush=True)	
user_ids = np.array(users["UserHash"])
user_embeddings = np.array(users["UserEmbedding"])
user_embeddings = np.array([np.fromstring(x, sep=" ") for x in user_embeddings]).reshape([len(user_ids), 100])
print(" Done.")

print("Normalizing users.. ", end='', flush=True)	
user_embeddings = np.apply_along_axis(lambda dimension: normalize_dimension(dimension), 0, user_embeddings)
print(" Done.")

# regressor = Regressor.LinearRegression

for part in user_recommendation_part:
	impression_count = 1.0
	click_count = 0.0
	total_impressions = 0.0
	local_clicks = 0.0
	local_count = 1.0
	users_to_update = list()
	clicks_to_update = list()
	user_recommendation_size = int(len(user_ids) * part)

	print("Starting evaluation of {0} with recommendation of size: {1} in count: {2}".format(algoName, part, user_recommendation_size))
	input = open("{0}//{1}//Processed//sorted_time_impressions.csv".format(path, name), "r")
	input.readline() # get rid of header
	line = input.readline()
	hour_begin_timestamp = datetime.datetime.fromtimestamp(int(line.split(",")[2])/1000)
	warmup = True
	
	algo = Regression(alpha, user_embeddings, user_ids, filter_clickers)
	recommended_users = list()
	for line in input:
		total_impressions += 1
		parts = line.split(",")
		user_id = int(parts[0])
		click = int(parts[1])
		timestamp_raw = int(parts[2])/1000
		timestamp = datetime.datetime.fromtimestamp(timestamp_raw)

		if warmup and (timestamp - hour_begin_timestamp).seconds < time_between_updates_in_seconds: 
			users_to_update.append(user_id)
			clicks_to_update.append(click)	
			continue	

		if (timestamp - hour_begin_timestamp).seconds >= time_between_updates_in_seconds and len(users_to_update) > 1000:
			recommended_users = algo.get_recommendations(user_recommendation_size)
			if len(clicks_to_update) != 0:
				algo.update(users_to_update, clicks_to_update)
			users_to_update = list()
			clicks_to_update = list()
			hour_begin_timestamp = timestamp
			warmup = False
			continue

		if user_id in recommended_users:	
			impression_count += 1
			local_count += 1
			click_count += click
			local_clicks += click
			users_to_update.append(user_id)
			clicks_to_update.append(click)	

		# print('.', end='', flush=True)	
		if total_impressions % 10000 == 0:
			print('{:.2%} Common Impressions: {} Cumulative CTR: {:.3%} CTR:{:.3%}'.format(total_impressions/total_lines, int(impression_count), click_count/impression_count, local_clicks/local_count) )
			local_clicks = 0.0	
			local_count = 1.0
			output.write("{0},{1},{2},{3},{4},{5},{6}\n".format(click_count, impression_count, total_impressions, algoName, part, user_recommendation_size, timestamp_raw))
			output.flush()
	input.close()
			

output.close()


















