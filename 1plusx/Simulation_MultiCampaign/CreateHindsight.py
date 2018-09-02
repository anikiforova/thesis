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

simulation_index = 2
total_lines = 16143123

campaign_ids = [597165, 837817, 722100]#, 809153]
algos = dict([])
ctr = dict([])
var = dict([])
calibration = dict([])
all_meta = Metadata(0)
for campaign_id in campaign_ids:
	print ("Starting building model for {}..".format(campaign_id))
	print("Reading impressions for campaign {}..".format(campaign_id))
	meta = Metadata(campaign_id)
	impressions_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)
	data = read_csv(impressions_file_path, ",")
	campaign_users 		 = data["UserHash"].values
	campaign_impressions = data["Click"].values
	
	print("Setting up regression for all users...")
	algo = Regression(all_meta)
	testMeta = TestMetadata(all_meta)
	testMeta.click_percent = 0.0
	algo.setup(testMeta)
	
	print("Fitting only campaign {} impressions...".format(campaign_id))
	#algo.update(campaign_users, campaign_impressions)
	prediction = np.array([algo.getPrediction(user_hash) for user_hash in campaign_users])

	ctr[campaign_id] = np.mean(campaign_impressions)
	var[campaign_id] = np.var(prediction)

	simulated_prediction = np.array([get_simulated_value(simulation_index, p, var[campaign_id], ctr[campaign_id]) for p in prediction ])
	simulated_prediction_ctr = np.mean(simulated_prediction) 
	calibration[campaign_id] = ctr[campaign_id] / simulated_prediction_ctr
	algos[campaign_id] = algo
	print("Campaign:{} Original CTR:{:.04} New CTR:{:.04} Calibration:{:.04}".format(campaign_id, ctr[campaign_id], simulated_prediction_ctr, calibration[campaign_id]))

input = open("{0}/sorted_time_impressions.csv".format(all_meta.path), "r")
input.readline()

output = open("{0}/multi_hindsight_sorted_time_impressions.csv".format(all_meta.path), "w")
output.write("UserHash,Timestamp,{}\n".format(",".join(str(c) for c in campaign_ids)))
print("Starting output to click file...")
for line_index, line in enumerate(input):
	line_parts = line.split(",")
	oritinal_campaign_id = int(line_parts[3])

	output_line = "{0},{1}".format(line_parts[0], line_parts[2])
	for campaign_id in campaign_ids:
		algo = algos[campaign_id]
		impression = line_parts[1]
		if campaign_id != oritinal_campaign_id:
			user_hash = int(line_parts[0])
			simulated_value = get_simulated_value(simulation_index, algo.getPrediction(user_hash), var[campaign_id], ctr[campaign_id]) 
			calibrated_simulated_value = simulated_value * calibration[campaign_id]
			impression = 0 if calibrated_simulated_value < np.random.uniform(0,1) else 1
		
		output_line = "{},{}".format(output_line, impression)

	output.write("{}\n".format(output_line))
	if line_index % 10000 == 0:
		print(".", end='', flush=True)	
		output.flush()
	
output.close()
#outputs[index].close()




























