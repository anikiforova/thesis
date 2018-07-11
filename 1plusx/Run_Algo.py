import numpy as np
import random 
import math
import time
import datetime
from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt

from Random import Random
from Regression import Regression
from LinUCB_Disjoint import LinUCB_Disjoint
from TS_Lin import TS_Lin
from GP_Clustered import GP_Clustered
from GP_FITC import GP_FITC
# from Random import Random

def normalize_dimension(dimension):
	min_value = np.min(dimension)
	max_value = np.max(dimension)
	return (dimension - min_value)/(max_value - min_value)

total_lines = 14392074

alphas 							= [0.02, 0.01, 0.001] #[0.001, 0.005, 0.01, 0.02, 0.05]
user_recommendation_part 		= [0.02]
hours 							= 1
time_between_updates_in_seconds = 60 * 60 * hours # 1 hour

print_output 		= True
print_mse 			= True
read_clustered_data = True
soft_click 			= False
filter_clickers 	= False


dimensions = 100
number_of_clusters = 10

clusters = "" # "_svd_" + str(dimensions)
algoName = "LinUCB_Disjoint"
algo_path = "{}_{}h_soft{}_filter{}{}".format(algoName, hours, soft_click, filter_clickers, clusters)

if print_output:
	output = open("./Results/{0}.csv".format(algoName), "a")
	#output.write("Clicks,Impressions,TotalImpressions,Method,RecommendationSizePercent,RecommendationSize,Timestamp,Alpha\n")
if print_mse:
	output_mse = open("./Results/MSE/{0}.csv".format(algoName), "w")
	output_mse.write("MSE,CumulativeMSE,TotalImpressions,Method,RecommendationSizePercent,RecommendationSize,Timestamp,Alpha\n")

algoName += clusters #+ "_f" + str(filter_clickers)

name = "809153"
path = "..//..//RawData//Campaigns"

print("Reading users.. ", end='', flush=True)	
users = read_csv("{0}//{1}//Processed//all_users{2}.csv".format(path, name, clusters), header=0)#, index_col=1)
print(" Done.")

print("Parsing users.. ", end='', flush=True)	
user_ids = np.array(users["UserHash"])
user_embeddings = np.array(users["UserEmbedding"])
user_embeddings = np.array([np.fromstring(x, sep=" ") for x in user_embeddings]).reshape([len(user_ids), dimensions])
print(" Done.")

print("Normalizing users .. ", end='', flush=True)	
user_embeddings = np.apply_along_axis(lambda dimension: normalize_dimension(dimension), 0, user_embeddings)
print(" Done.")

print("Reading and normalizing cluster data.. ", end='', flush=True)
cluster_embeddings = []
if read_clustered_data:
	clusters = read_csv("{0}//{1}//Processed//all_users_clusters_{2}.csv".format(path, name, number_of_clusters), header=0)
	cluster_embeddings = np.array(clusters["Cluster"])
	cluster_embeddings = np.array([np.fromstring(x, sep=" ") for x in cluster_embeddings]).reshape([number_of_clusters, dimensions])
	cluster_embeddings 	= np.apply_along_axis(lambda dimension: normalize_dimension(dimension), 0, cluster_embeddings)

	# clustered_users = read_csv("{0}//{1}//Processed//all_users_clustered_{2}.csv".format(path, name, number_of_clusters), header=0)
	# clustered_user_embeddings = np.array(clustered_users["UserEmbedding"])
	# clustered_user_embeddings = np.array([np.fromstring(x, sep=" ") for x in clustered_user_embeddings]).reshape([len(user_ids), number_of_clusters])
	# clustered_user_embeddings = np.apply_along_axis(lambda dimension: normalize_dimension(dimension), 0, clustered_user_embeddings)
print(" Done.")

# regressor = Regressor.LinearRegression

algo = TS_FITC(user_embeddings, user_ids, cluster_embeddings, dimensions, filter_clickers, soft_click)

for alpha in alphas:
	for part in user_recommendation_part:
		impression_count = 1.0
		click_count = 0.0
		total_impressions = 0.0
		local_clicks = 0.0
		local_count = 1.0
		missed_clicks = 0.0
		total_clicks = 0.0
		local_missed_clicks = 0.0
		total_local_clicks = 0.0
		impressions_per_recommendation_group = 0.0

		SE = 0.0
		cumulative_SE = 0.0 

		users_to_update = list()
		clicks_to_update = list()
		user_recommendation_size = int(len(user_ids) * part)

		print("Starting evaluation of {0} with recommendation of size: {1}% in count: {2} and alpha: {3}".format(algoName, part * 100, user_recommendation_size, alpha))
		input = open("{0}//{1}//Processed//sorted_time_impressions.csv".format(path, name), "r")
		input.readline() # get rid of header
		line = input.readline()
		hour_begin_timestamp = datetime.datetime.fromtimestamp(int(line.split(",")[2])/1000)
		warmup = True
		
		# algo = Regression(alpha, user_embeddings, user_ids, cluster_embeddings, filter_clickers, soft_click)
		algo.setup(alpha)

		recommended_users = list()
		for line in input:
			total_impressions += 1
			parts = line.split(",")
			user_id = int(parts[0])
			click = int(parts[1])
			timestamp_raw = int(parts[2])/1000
			timestamp = datetime.datetime.fromtimestamp(timestamp_raw)
			
			impressions_per_recommendation_group += 1
			cur_SE 			= (click - algo.getPrediction(user_id)) ** 2
			SE 				+= cur_SE 
			cumulative_SE 	+= cur_SE

			if warmup and (timestamp - hour_begin_timestamp).seconds < time_between_updates_in_seconds: 
				users_to_update.append(user_id)
				clicks_to_update.append(click)	
				continue	

			if (timestamp - hour_begin_timestamp).seconds >= time_between_updates_in_seconds and len(users_to_update) > 1000:
				recommended_users = algo.get_recommendations(user_recommendation_size)
				if len(clicks_to_update) != 0:
					if print_mse:
						MSE = SE / impressions_per_recommendation_group
						cumulative_MSE = cumulative_SE / total_impressions
						output_mse.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(MSE, cumulative_MSE, total_impressions, algoName, part, user_recommendation_size, timestamp_raw, alpha))
						output_mse.flush()
					SE = 0.0
					impressions_per_recommendation_group = 0.0
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
			else:
				missed_clicks += click
				local_missed_clicks += click

			total_clicks += click
			total_local_clicks += click

			# print('.', end='', flush=True)	
			if total_impressions % 10000 == 0:
				print('{:.2%} Common Impressions: {} Cumulative CTR: {:.3%} CTR:{:.3%} Cumulative MC: {:.3%} MC:{:.3%}'.format(total_impressions/total_lines, int(impression_count), click_count/impression_count, local_clicks/local_count, missed_clicks/total_clicks, local_missed_clicks/total_local_clicks))

				if print_output:
					output.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(click_count, impression_count, total_impressions, algoName, part, user_recommendation_size, timestamp_raw, alpha))
					output.flush()

				local_clicks = 0.0	
				local_count = 1.0
				local_missed_clicks = 0.0
				total_local_clicks = 0.0
				SE = 0.0
				
				
		input.close()
				
if print_output:
	output.close()


















