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

import MetricsCalculator
import TestBuilder
import Util

campaign_id = 809153 # 837817# 809153 # 597165 # 837817
meta = Metadata(campaign_id)

simulated = False
simulation_id = 0

algoName = "LinUCB_Disjoint"
algo = Regression(meta)

testsMeta = TestBuilder.get_lin_tests_mini(meta, 4)

output_path = "./Results/{0}/{1}{2}_Metrics.csv".format(meta.campaign_id, algoName)
if simulated:
	output_path = "./Results/{0}/Simulated/{1}/{2}.csv".format(meta.campaign_id, simulation_id, algoName)

output_column_names = False
if not Path(output_path).is_file():
	output = open(output_path, "w")	
	output_column_names = True;
else:
	output = open(output_path, "a")

# specials = read_csv("{0}//special_users.csv".format(meta.path), header=0)
# specials = set(specials["UserHash"].values)
# specials_count = len(specials)

for testMeta in testsMeta:
	algo.setup(testMeta)

	if output_column_names:
		output.write("Clicks,Impressions,TotalImpressions,Method,Timestamp,BatchCTR,ModelCTR,MSE,MMSE,FullMSE,FullROC,FullTPR,FullFPR,FullFNR,FullPPR,ModelCalibration,ModelNE,ModelRIG{}\n".format(testMeta.get_additional_column_names()))
		output_column_names = False

	warmup = True

	all_model_impressions = list()
	all_prediction_values = list()
	all_prediction_clicks = list()
	all_impressions = list()
	batch_users = list()
	batch_clicks = list()

	recommended_users = list()

	print("Starting evaluation of {} with {}".format(algoName, testMeta.get_additional_info()))

	file_name = "{0}/sorted_time_impressions.csv".format(meta.path)
	if simulated:
		file_name = "{0}/simulated_time_impressions_s{1}.csv".format(meta.path, simulation_id)
	
	input = open(file_name, "r")

	input.readline() # get rid of header
	_, _, _, hour_begin_timestamp = Util.get_line_info(input.readline())
	
	print("Time between :{0}".format(testMeta.get_time_between_updates_in_seconds()))
	for total_impressions, line in enumerate(input):
		user_id, click, timestamp_raw, timestamp = Util.get_line_info(line)
		
		if warmup and (timestamp - hour_begin_timestamp).seconds < testMeta.get_time_between_updates_in_seconds(): 
			batch_users.append(user_id)
			batch_clicks.append(click)
			continue	

		if (timestamp - hour_begin_timestamp).seconds >= testMeta.get_time_between_updates_in_seconds():			
			algo.update(batch_users, batch_clicks)
			recommended_users = algo.get_recommendations(testMeta.recommendation_part)

			batch_users = list()
			batch_clicks = list()
			
			hour_begin_timestamp = timestamp
			warmup = False
			continue

		all_impressions.append(click)
		all_prediction_values.append(algo.getPrediction(user_id))
		if user_id in recommended_users:	
			batch_users.append(user_id)
			batch_clicks.append(click)
			all_model_impressions.append(click)
			all_prediction_clicks.append(click)
		else:
			all_prediction_clicks.append(0)

		# print('.', end='', flush=True)	
		if total_impressions % 10000 == 0:
			iterative_metrics = MetricsCalculator.get_iterative_model_metrics(all_impressions, all_prediction_values,all_model_impressions, batch_clicks)
			full_metrics 	= MetricsCalculator.get_full_model_metrics(all_impressions, all_prediction_clicks)
			entropy_metrics = MetricsCalculator.get_entropy_metrics(all_prediction_clicks, all_prediction_values, np.mean(all_impressions))

			print('{:.2%} Clicks:{} Imp:{} BatchCTR:{:.3%} CumCTR:{:.3%} CumMSE:{:.03} CumMMSE:{:.03} TPR:{:.03}'.format(
				total_impressions/meta.total_lines[campaign_id], 
				iterative_metrics["Clicks"],
				iterative_metrics["Impressions"],
				iterative_metrics["ModelBatchCTR"],
				iterative_metrics["ModelCTR"],
				iterative_metrics["MSE"],
				iterative_metrics["MMSE"],
				full_metrics["TPR"]))

			output.write("{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}\n".format(
				iterative_metrics["Clicks"], 
				iterative_metrics["Impressions"], 
				total_impressions, 
				algoName, 
				timestamp_raw, 
				iterative_metrics["ModelBatchCTR"], 
				iterative_metrics["ModelCTR"], 
				iterative_metrics["MSE"], 
				iterative_metrics["MMSE"], 
				full_metrics["MSE"],
				full_metrics["ROC"],
				full_metrics["TPR"],
				full_metrics["FPR"],
				full_metrics["FNR"],
				full_metrics["PPR"],
				entropy_metrics["Calibration"],
				entropy_metrics["NE"],
				entropy_metrics["RIG"],
				testMeta.get_additional_column_info()))
			output.flush()
			
	input.close()
				
output.close()


















