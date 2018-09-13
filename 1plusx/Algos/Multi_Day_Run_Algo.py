import numpy as np
import random 
import math
import time
import datetime
import pandas as pd
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

campaign_ids = set([866128, 856805, 847460, 858140, 865041])
campaign_ids_str = ",".join([str(x) for x in campaign_ids])

# this runs a one campaign LinUCB on multiple campaigns at the same time to see how we can estimate click users
meta = Metadata("LinUCB_Disjoint", campaign_id = 5, initialize_user_embeddings = False)
algo = LinUCB_Disjoint(meta)

testsMeta = TestBuilder.get_lin_test(meta, 6)

output_path = "./Results/{0}/{1}_Metrics2.csv".format(meta.campaign_id, meta.algo_name)
output_campaign_log_path = "./Log/{0}/Campaign_Log.csv".format(meta.campaign_id, meta.algo_name)

output_column_names = False

if not Path(output_path).is_file():
	output = open(output_path, "w")	
	output_column_names = True;
else:
	output = open(output_path, "a")

output_campaign_log = open(output_campaign_log_path, "a")	

for testMeta in testsMeta:
	algo.setup(testMeta)

	if output_column_names:
		#output_campaign_log.write("CampaignId,Clicks,Impressions,{}\n".format(testMeta.get_algo_column_names()))
		output.write("Clicks,Impressions,TotalImpressions,Timestamp,BatchCTR,ModelCTR,MSE,MMSE,FullMSE,FullROC,FullTPR,FullFPR,FullFNR,FullPPR,ModelCalibration,ModelNE,ModelRIG,{}\n".format(testMeta.get_algo_column_names()))
		output_column_names = False

	all_model_impressions = list()
	all_prediction_values = list()
	all_prediction_clicks = list()
	all_impressions = list()
	batch_users = list()
	batch_clicks = list()

	impressions_per_campaign = dict(zip(campaign_ids, np.zeros(len(campaign_ids))))
	clicks_per_campaign = dict(zip(campaign_ids, np.zeros(len(campaign_ids))))

	recommended_users = list()

	print("Starting evaluation of {}".format(testMeta.get_algo_info()))
	
	days = pd.date_range(start='15/8/2018', end='17/08/2018')

	total_impressions = 0
	warmup = True # it's important it stays outside of the day loop, so it does warmup only one time.
	for date in days:
		print("Starting {}..".format(date))
		date = date.strftime("%Y-%m-%d")

		algo.update_user_embeddings("_" + date)
		algo.reset_predictions()
		if not warmup:
			algo.update_prediction()
			recommended_users = algo.get_recommendations(testMeta.recommendation_part)

		file_name = "{0}/sorted_time_impressions_{1}.csv".format(meta.path, date)
		
		input = open(file_name, "r")
		input.readline() # get rid of header
		_, _, _, _, hour_begin_timestamp = Util.get_campaign_line_info(input.readline())
		
		#print("Time between updates:{0}".format(testMeta.get_time_between_updates_in_seconds()))
		for cur_file_impressions, line in enumerate(input):
			total_impressions += 1
			displayed_campaign, user_id, click, timestamp_raw, timestamp = Util.get_campaign_line_info(line)
			
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

				impressions_per_campaign[displayed_campaign] += 1
				clicks_per_campaign[displayed_campaign] += click
			else:
				all_prediction_clicks.append(0)

			# print('.', end='', flush=True)	
			if total_impressions % 100000 == 0:
				iterative_metrics = MetricsCalculator.get_iterative_model_metrics(all_impressions, all_prediction_values,all_model_impressions, batch_clicks)
				full_metrics 	= MetricsCalculator.get_full_model_metrics(all_impressions, all_prediction_clicks)
				entropy_metrics = MetricsCalculator.get_entropy_metrics(all_prediction_clicks, all_prediction_values, np.mean(all_impressions))

				print('{} {:.2%} Clicks:{} Imp:{} BatchCTR:{:.3%} CumCTR:{:.3%} CumMSE:{:.03} CumMMSE:{:.03} TPR:{:.03}'.format(
					date,
					cur_file_impressions/meta.total_lines[meta.campaign_id][date], 
					iterative_metrics["Clicks"],
					iterative_metrics["Impressions"],
					iterative_metrics["ModelBatchCTR"],
					iterative_metrics["ModelCTR"],
					iterative_metrics["MSE"],
					iterative_metrics["MMSE"],
					full_metrics["TPR"]))

				output.write("{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}\n".format(
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
		algo.update(batch_users, batch_clicks)
		batch_users = list()
		batch_clicks = list()
				
	for campaign_id in campaign_ids:
		output_campaign_log.write("{},{},{},{}\n".format(campaign_id, impressions_per_campaign[campaign_id], clicks_per_campaign[campaign_id], testMeta.get_algo_column_info()))

output.close()
output_campaign_log.close()


















