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

import TestBuilder
import Util
import MetricsCalculator

detailed = False
campaign_id = 837817
algoName = "LinUCB_Disjoint"
meta = Metadata(algoName, campaign_id)
algo = LinUCB_Disjoint(meta)

testsMeta = TestBuilder.get_lin_tests_mini(meta)

ctr_multipliers = [0.5, 1, 2, 5, 10]
# ctr_multipliers = [1]
simulation_ids = [2]

output_path = "./Results/{0}/Simulated/{1}_2.csv".format(meta.campaign_id, algoName)
output_column_names = False
if not Path(output_path).is_file():
	output = open(output_path, "w")	
	output_column_names = True;
else:
	output = open(output_path, "a")

for simulation_id in simulation_ids:
	for multiplier in ctr_multipliers:
	
		print("Starting simulation:{} with multiplier {}".format(simulation_id, multiplier))
	
		for testMeta in testsMeta:
			algo.setup(testMeta)

			if output_column_names:
				output.write("SimulationId,CTRMultiplier,Clicks,Impressions,TotalImpressions,Timestamp,BatchCTR,ModelCTR,MSE,MMSE,FullMSE,FullROC,FullTPR,FullFPR,FullFNR,FullPPR,ModelCalibration,ModelNE,ModelRIG,{}\n".format(testMeta.get_algo_column_names()))
				output_column_names = False

			warmup = True

			all_model_impressions = list()
			all_prediction_values = list()
			all_prediction_clicks = list()
			all_impressions = list()
			batch_users = list()
			batch_clicks = list()

			recommended_users = list()

			print("Starting evaluation of {}".format(testMeta.get_algo_info()))

			file_name = "{0}//simulated_time_impressions_s_{1}_m_{2}.csv".format(meta.path, simulation_id, multiplier)
			# file_name = "{0}//simulated_time_impressions_s{1}.csv".format(meta.path, simulation_id)
			
			input = open(file_name, "r")

			input.readline() # get rid of header
			_, _, _, hour_begin_timestamp = Util.get_line_info(input.readline())
			
			print("Time between updates:{0}".format(testMeta.get_time_between_updates_in_seconds()))
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
					full_metrics 	= MetricsCalculator.get_full_model_metrics(all_impressions, all_prediction_clicks, detailed = detailed)
					entropy_metrics = MetricsCalculator.get_entropy_metrics(all_prediction_clicks, all_prediction_values, np.mean(all_impressions), detailed = detailed)

					print('{:.2%} Clicks:{} Imp:{} BatchCTR:{:.3%} CumCTR:{:.3%} CumMSE:{:.03} CumMMSE:{:.03} TPR:{:.03}'.format(
						total_impressions/meta.total_lines[campaign_id], 
						iterative_metrics["Clicks"],
						iterative_metrics["Impressions"],
						iterative_metrics["ModelBatchCTR"],
						iterative_metrics["ModelCTR"],
						iterative_metrics["MSE"],
						iterative_metrics["MMSE"],
						full_metrics["TPR"]))


					output.write("{},{},{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}\n".format(
						simulation_id,
						multiplier,
						iterative_metrics["Clicks"], 
						iterative_metrics["Impressions"], 
						total_impressions, 
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
						testMeta.get_algo_column_info()))
					output.flush()
					
			input.close()
				
						
output.close()

















