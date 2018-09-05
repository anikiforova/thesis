import pandas as pd
import os 
import glob
import math
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv

import sys
from os import path
# to be able to access sister folders
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import Algos.Util as Util
import Algos.MetricsCalculator as Metrics
from Algos.AlgoBase import AlgoBase
from Algos.Metadata import Metadata 
from Algos.Regression import Regression 
from Algos.TestMetadata import TestMetadata 

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def square_func(x):
	a = 1
	b = -1
	c = 0.5
	return a*x*x + b*x + c

def normalize(l):
	min_val = np.min(l)
	max_val = np.max(l)

	return np.array([(v - min_val)/(max_val - min_val) for v in l])

def get_simulated_value(simulation_index, prediction, var, ctr):
	simulated_value = 0.0
	if simulation_index == 0:
		simulated_value = prediction + np.random.uniform(-ctr/10, +ctr/10, 1)
	elif simulation_index == 1:
		simulated_value = np.random.normal(prediction, var, 1)
	elif simulation_index == 2:
		simulated_value = np.random.normal(prediction, var, 1) + np.random.uniform(-ctr/10, +ctr/10, 1)
	elif simulation_index == 3:
		simulated_value = sigmoid(prediction) + np.random.uniform(-ctr/10, +ctr/10, 1)
	elif simulation_index == 4:
		simulated_value = sigmoid(square_func(prediction)) + np.random.uniform(-ctr/10, +ctr/10, 1)
	return simulated_value

path = "../../RawData/Campaigns/"
impressions_file_path_extension = "/Processed/sorted_time_impressions.csv"
hindsight_multi_campaign_path = "../../RawData/Multi-Campaign/Processed/SimulationHindsight"
real_multi_campaign_path = "../../RawData/Multi-Campaign/Processed/FiveCampaigns"

simulation_index = 2

campaign_ids = [866128, 856805, 847460, 858140, 865041]#, 809153]
algos = dict([])
ctr = dict([])
var = dict([])
calibration = dict([])

for campaign_id in campaign_ids:
	print ("Starting building model for {}..".format(campaign_id))
	print("Reading impressions for campaign {}..".format(campaign_id))
	meta = Metadata(campaign_id)
	impressions_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)
	data = read_csv(impressions_file_path, ",")
	campaign_users 		 = data["UserHash"].values
	campaign_impressions = data["Click"].values
	
	print("Setting up regression for all users...")
	algo = Regression(meta)
	testMeta = TestMetadata(meta)
	testMeta.click_percent = 0.0
	algo.setup(testMeta)
	
	print("Fitting only campaign {} impressions...".format(campaign_id))
	algo.update(campaign_users, campaign_impressions)
	prediction = np.array([algo.getPrediction(user_hash) for user_hash in campaign_users])

	ctr[campaign_id] = np.mean(campaign_impressions)
	var[campaign_id] = np.var(prediction)

	simulated_prediction = np.array([get_simulated_value(simulation_index, p, var[campaign_id], ctr[campaign_id]) for p in prediction ])
	simulated_prediction_ctr = np.mean(simulated_prediction) 
	calibration[campaign_id] = ctr[campaign_id] / simulated_prediction_ctr
	algos[campaign_id] = algo
	print("Campaign:{} Original CTR:{:.04} New CTR:{:.04} Calibration:{:.04}".format(campaign_id, ctr[campaign_id], simulated_prediction_ctr, calibration[campaign_id]))

a = pd.date_range(start='15/8/2018', end='28/08/2018')

printHeader = True
for date in a:
	date = date.strftime("%Y-%m-%d")
	print("\nDate {}...".format(date))

	input = open("{0}/sorted_time_impressions_{1}.csv".format(real_multi_campaign_path, date), "r")
	input.readline()

	output = open("{0}/sorted_time_impressions_{1}.csv".format(hindsight_multi_campaign_path, date), "w")
	output.write("UserHash,Timestamp,{}\n".format(",".join(str(c) for c in campaign_ids)))
	
	user_ids, user_embeddings = Metadata.read_user_embeddings_by_path("{0}/all_users_{1}.csv".format(real_multi_campaign_path, date))
	dictionary = dict(zip(user_ids, user_embeddings))
	random_values = np.random.uniform(0, 1, 10000 * len(campaign_ids)).reshape([10000, len(campaign_ids)])
	for line_index, line in enumerate(input):
		line_parts = line.split(",")
		oritinal_campaign_id = int(line_parts[0])
		user_hash = int(line_parts[1])
		click = line_parts[2]
		timestamp = line_parts[3][0:-1]

		results = list()
		for campaign_index, campaign_id in enumerate(campaign_ids):
			algo = algos[campaign_id]
			if campaign_id != oritinal_campaign_id:
				user_embedding = dictionary[user_hash]
				predicted_value = algo.predict_now(user_embedding)

				simulated_value = get_simulated_value(simulation_index, predicted_value, var[campaign_id], ctr[campaign_id]) 
				calibrated_simulated_value = simulated_value * calibration[campaign_id]
				click = "0" if calibrated_simulated_value < random_values[line_index%10000][campaign_index] else "1"
			results.append(click)

		output.write("{0},{1},{2}\n".format(user_hash, timestamp, ",".join(results)))
		if line_index % 10000 == 0:
			random_values = np.random.uniform(0, 1, 10000 * len(campaign_ids)).reshape([10000, len(campaign_ids)])
			print(".", end='', flush=True)	
			output.flush()
		
	output.close()
#outputs[index].close()




























