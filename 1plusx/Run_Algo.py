import numpy as np
import random 
import math
import time
from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt

from Regression import Regression
# from Random import Random

total_lines = 7196037
repetitions = 10

impressions_between_update = 100

alpha = 0.0
algoName = "Regression"
warmup_impression_count = 1000

output = open("./Results/{0}.csv".format(algoName), "w")
output.write("Clicks,Impressions,TotalImpressions,Method,RecommendationSizePercent,RecommendationSize\n")

name = "809153"
path = "..//..//RawData//Campaigns"
users = read_csv("{0}//{1}//Processed//all_users.csv".format(path, name), header=0)#, index_col=1)
user_ids = np.array(users["UserHash"])
user_embeddings = np.array(users["UserEmbedding"])
user_embeddings = np.array([np.fromstring(x, sep=" ") for x in user_embeddings]).reshape([len(user_ids), 100])


# regressor = Regressor.LinearRegression

user_recommendation_part = np.arange(0.1, 0.2, 0.1)

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
	input = open("{0}//{1}//Processed//impressions.csv".format(path, name), "r")
	input.readline() # get rid of header

	algo = Regression(alpha, user_embeddings, user_ids)
	recommended_users = list()
	for line in input:
		total_impressions += 1
		parts = line.split(",")
		user_id = int(parts[0])
		click = int(parts[1])

		if total_impressions < warmup_impression_count:
			users_to_update.append(user_id)
			clicks_to_update.append(click)	
			continue	

		if total_impressions == warmup_impression_count:
			algo.update(users_to_update, clicks_to_update)
			recommended_users = algo.get_recommendations(user_recommendation_size)
			users_to_update = list()
			clicks_to_update = list()
			continue

		if user_id in recommended_users:
			# print('.', end='', flush=True)	
			impression_count += 1
			local_count += 1
			click_count += click
			local_clicks += click
			users_to_update.append(user_id)
			clicks_to_update.append(click)

			if impression_count % impressions_between_update == 0:
				recommended_users = algo.get_recommendations(user_recommendation_size)
				algo.update(users_to_update, clicks_to_update)
				users_to_update = list()
				clicks_to_update = list()

		# print('.', end='', flush=True)	
		if total_impressions % 10000 == 0:
			print('{:.2%} Cumulative CTR: {:.3%} CTR:{:.3%}'.format(total_impressions/total_lines, click_count/impression_count, local_clicks/local_count) )
			local_clicks = 0.0	
			local_count = 1.0
			output.write("{0},{1},{2},{3},{4},{5}\n".format(click_count, impression_count, total_impressions, algoName, part, user_recommendation_size))
			output.flush()
	input.close()
			

output.close()


















