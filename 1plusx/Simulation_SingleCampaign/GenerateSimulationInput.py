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
simulated_impressions_file_path_extension = "/Processed/simulated_time_impressions"
campaign_id = 837817 
print ("Starting model for {}..".format(campaign_id))
impressions_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)

data = read_csv(impressions_file_path, ",")

users 		= data["UserHash"].values
impressions = data["Click"].values

meta = Metadata("Regression", campaign_id)
algo = Regression(meta)
testMeta = TestMetadata(meta)
testMeta.click_percent = 0.0
algo.setup(testMeta)
algo.update(users, impressions)
prediction = np.array([algo.getPrediction(user_hash) for user_hash in users])
ctr = np.mean(impressions)
var = np.var(prediction)
total_count = len(impressions)

print("CTR:{:.03} VAR: {:.06} Total:{}".format(ctr, var, total_count))
print("Starting model evaluation..")

input_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)
partial_output_file_path = "{0}/{1}/{2}".format(path, campaign_id, simulated_impressions_file_path_extension)

output_stats = open("./Results/{0}/Simulated/CalibrationDetails.csv".format(campaign_id), "w")
output_stats.write("CampaignId,SimulationId,Cutoff,CTR,MSE,Calibration,NE,RIG,TPR,FPR,FNR,PPR,ROC\n")

ctr_multipliers = [0.1, 0.5, 1, 5]
simulation_ids = [2, 4]

for id in simulation_ids:
	print("Starting simulation:{}".format(id))
	simulation_values = np.array([get_simulated_value(id, pred, var, ctr) for pred in prediction])
	for multiplier in ctr_multipliers:		
		print("Starting simulation:{} with multiplier {} - {}".format(id, multiplier, multiplier * ctr))
		cutoff_value = np.percentile(simulation_values, 100 - multiplier * ctr * 100)
		simulated_impressions = np.array(simulation_values > cutoff_value, dtype=int)
		
		print("Starting writing..")
		with open("{0}_s_{1}_m_{2}.csv".format(partial_output_file_path, id, multiplier), "w") as output:
			with open(input_file_path, "r") as input:
			
				header = input.readline() 
				output.write(header)
				
				for index, line in enumerate(input):
					user_hash, _, timestamp_raw, _ = Util.get_line_info(line) 
					output.write("{},{},{}\n".format(user_hash, simulated_impressions[index][0], timestamp_raw))
		

# for id in simulation_ids:
# 	print("Starting simulation:{}".format(id))
# 	simulation_values = np.array([get_simulated_value(id, pred, var, ctr) for pred in prediction])
# 	for multiplier in ctr_multipliers:		
# 		print("Starting simulation:{} with multiplier {}".format(id, multiplier))
# 		cutoff_value = np.percentile(simulation_values, 100 - multiplier * ctr * 100)
# 		simulated_impressions = np.array(simulation_values > cutoff_value, dtype=int)

# 		metrics = Metrics.get_full_model_metrics(impressions, simulated_impressions)
# 		entropy_metrics = Metrics.get_entropy_metrics(simulated_impressions, simulation_values, ctr)

# 		output_stats.write("{},{},{:.03},{:.03},{:.03},{:.03},{:.03},{:.03},{:.03},{:.03},{:.03},{:.03},{:.03}\n".format(campaign_id, 
# 			id, 
# 			cutoff_value, 
# 			entropy_metrics["CTR"], 
# 			metrics["MSE"], 
# 			entropy_metrics["Calibration"], 
# 			entropy_metrics["NE"],
# 			entropy_metrics["RIG"], 
# 			metrics["TPR"], 
# 			metrics["FPR"], 
# 			metrics["FNR"], 
# 			metrics["PPR"], 
# 			metrics["ROC"]))
	
# output_stats.close()





























