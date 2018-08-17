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
simulated_impressions_file_path_extension = "/Processed/simulated_time_impressions_s"
campaign_id = 837817 #  597165 #
print ("Starting model for {}..".format(campaign_id))
impressions_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)

data = read_csv(impressions_file_path, ",")

users 		= data["UserHash"].values
impressions = data["Click"].values

meta = Metadata(campaign_id)
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

input = open("{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension), "r")
header = input.readline() # get rid of header

outputs = list()
simulation_values = np.array([])
simulation_count = 5

min_val = np.min(prediction)
max_val = np.max(prediction)

for i in np.arange(0, simulation_count):
	cur_values = np.array([get_simulated_value(i, pred, var, ctr) for pred in prediction])
	simulation_values = np.append(simulation_values, cur_values)

	# output = open("{0}/{1}/{2}{3}.csv".format(path, campaign_id, simulated_impressions_file_path_extension, i), "w")
	# output.write(header)
	# outputs.append(output)

simulation_values = simulation_values.reshape([simulation_count, total_count])

# print("Starting output...")
# for index, line in enumerate(input):
# 	user_hash, click, timestamp_raw, timestamp = Util.get_line_info(line) 
# 	for output_index in np.arange(0, simulation_count):
# 		outputs[output_index].write("{},{},{}\n".format(user_hash, int(simulation_clicks[output_index][index]), timestamp_raw))
# input.close()
# print("Done with output..")

ctr_multipliers = [1, 2, 5, 10, 20, 50, 100, 200, 99.0/ctr]

output_stats = open("./Results/{0}/Simulated/SimulationDetails.csv".format(campaign_id), "w")
output_stats.write("CampaignId,SimulationId,Cutoff,CTR,MSE,Calibration,NE,RIG,TPR,FPR,FNR,PPR,ROC\n")

for index in np.arange(0, simulation_count):
	for multiplier in ctr_multipliers:
		cutoff_value = np.percentile(simulation_values[index], 100 - multiplier * ctr * 100)
		print("Simulation:{} Multiplier:{} CTR:{:.04} Cutoff:{:.04}".format(i, multiplier, multiplier * ctr, cutoff_value))

		simulation_impressions = np.array(simulation_values[index] > cutoff_value, dtype=int)
		metrics = Metrics.get_full_model_metrics(impressions, simulation_impressions)
		
		cur_ctr, Calibration, NE, RIG = Metrics.get_entropy_metrics(simulation_impressions, simulation_values[index], ctr)

		output_stats.write("{},{},{:.04},{:.04},{:.04},{:.04},{:.04},{:.04},{:.04},{:.04},{:.04},{:.04},{:.04}\n".format(campaign_id, index, cutoff_value, cur_ctr, metrics["MSE"], Calibration, NE, RIG, metrics["TPR"], metrics["FPR"], metrics["FNR"], metrics["PPR"], metrics["ROC"]))
	
output_stats.close()
#outputs[index].close()




























