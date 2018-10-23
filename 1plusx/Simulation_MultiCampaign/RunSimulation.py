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
from termcolor import colored
from random import randint

import sys
from os import path
# to be able to access sister folders
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from Algos.Metadata import Metadata
from Algos.TestMetadata import TestMetadata

from Algos.Random_Multi import Random_Multi
from Algos.LinUCB_Disjoint_Multi import LinUCB_Disjoint_Multi

from Algos.SimulationType import get_friendly_name

from Algos.MetricsCalculator import get_basic_metrics    
from Algos.TestBuilder import get_simulation_lin_multi_target_test
from Algos.TestBuilder import get_simulations_multi_test
from Algos.Util import get_campaign_line_info

from SimulationController import SimulationController

campaign_ids = np.array([866128, 856805, 847460, 858140, 865041])
campaign_ids_str = ",".join([str(x) for x in campaign_ids])

meta = Metadata("LinUCB_Disjoint_Multi_Target", campaign_id = 5, initialize_user_embeddings = False)
days = pd.date_range(start='15/8/2018', end='20/08/2018') 

algo = LinUCB_Disjoint_Multi(meta, campaign_ids, days[0], days[-1]+ 1)

simulation_controller = SimulationController(reinitialize_calibration = False)

testsMeta = get_simulations_multi_test(meta, 6)

output_path = "./Results/{0}/Simulation/{1}_All.csv".format(meta.campaign_id, meta.algo_name)
output_log_path = "./Log/{0}/Simulation/{1}_All.csv".format(meta.campaign_id, meta.algo_name)
output_campaign_log_path = "./Log/{0}/Simulation/Campaign_Log_All.csv".format(meta.campaign_id)

output_column_names = False
if not Path(output_path).is_file():
	output = open(output_path, "w")	
	log_output = open(output_log_path, "w")	
	output_column_names = True;
else:
	output = open(output_path, "a")
	log_output = open(output_log_path, "a")	

output_campaign_log = open(output_campaign_log_path, "a")	

for index, testMeta in enumerate(testsMeta):
	algo.setup(testMeta)
	simulation_controller.setup(testMeta)
	
	simulation_type_friendly_name = get_friendly_name(testMeta.simulation_type)

	if output_column_names:
		algo_column_names = testMeta.get_algo_column_names()
		output_campaign_log.write("CampaignId,Clicks,Impressions,{}\n".format(algo_column_names))
		log_output.write("Type,Timestamp,TotalImpressions,{},{}\n".format(campaign_ids_str, algo_column_names))
		output.write("Clicks,Impressions,TotalImpressions,Timestamp,BatchCTR,FullCTR,BatchMSE,BatchMMSE,FullMSE,FullMMSE,{}\n".format(algo_column_names))
		output_column_names = False

	all_model_impressions		= list()
	all_model_prediction_values = list()

	cur_model_impressions		= list()
	cur_model_prediction_values	= list()

	batch_users					= list()
	batch_clicks				= list()
	batch_campaign_ids			= list()

	impressions_per_campaign = dict(zip(campaign_ids, np.zeros(len(campaign_ids))))
	clicks_per_campaign = dict(zip(campaign_ids, np.zeros(len(campaign_ids))))

	print(colored("{}/{} Starting evaluation of  {}".format(index + 1, len(testsMeta), testMeta.get_algo_info()), "green"))

	total_impressions = 0
	warmup = True # it's important it stays outside of the day loop, so it does warmup only one time.
	# print("Remaining target budgets:")
	# print(algo.get_remaining_target_budgets())
	for date in days:
		print(colored("Starting {}..".format(date), "green"))
		date = date.strftime("%Y-%m-%d")

		file_name = "{0}/sorted_time_impressions_{1}.csv".format(meta.path, date)
		
		input = open(file_name, "r")
		header = input.readline() # get rid of header
		_, _, _, hour_begin_timestamp_raw, hour_begin_timestamp = get_campaign_line_info(input.readline())
		
		algo.reset_expected_impression_count(hour_begin_timestamp_raw)
		algo.start_new_day()
		algo.update_user_embeddings("_" + date)
		algo.reset_local_target_budgets()
		algo.reset_predictions()
		algo.update_multi_campaign_predictions(campaign_ids)
		
		#print("Time between updates:{0}".format(testMeta.get_time_between_updates_in_seconds()))
		for cur_file_impressions, line in enumerate(input):
			total_impressions += 1
			displayed_campaign, user_id, click, timestamp_raw, timestamp = get_campaign_line_info(line)
			
			if warmup and (timestamp - hour_begin_timestamp).seconds < testMeta.get_time_between_updates_in_seconds():
				user_embedding = algo.get_user_embedding(user_id)
				simulated_click = simulation_controller.get_simulated_impression(displayed_campaign, user_embedding)

				batch_campaign_ids.append(displayed_campaign)
				batch_users.append(user_id)
				batch_clicks.append(simulated_click)
				continue	

			if (timestamp - hour_begin_timestamp).seconds >= testMeta.get_time_between_updates_in_seconds():
				algo.reset_expected_impression_count(timestamp_raw)
				algo.update(batch_campaign_ids, batch_users, batch_clicks, timestamp_raw)
				algo.log_budgets(log_output, total_impressions, timestamp_raw)

				batch_users, batch_clicks, batch_campaign_ids = list(), list(), list()
				
				hour_begin_timestamp = timestamp
				warmup = False
				continue

			assigned_campaign = algo.getAssignment(user_id)
			# if assigned_campaign == displayed_campaign:	
			user_embedding = algo.get_user_embedding(user_id)
			simulated_click = simulation_controller.get_simulated_impression(assigned_campaign, user_embedding)

			batch_campaign_ids.append(assigned_campaign)
			batch_users.append(user_id)
			batch_clicks.append(simulated_click)

			all_model_impressions.append(simulated_click)
			cur_model_impressions.append(simulated_click)
			
			prediction = algo.getPrediction(user_id)
			all_model_prediction_values.append(prediction)
			cur_model_prediction_values.append(prediction)
			
			budget_is_exhausted = algo.consume_campaign_budget(assigned_campaign)
			if testMeta.early_update and budget_is_exhausted:
				print(colored("Exhausted budget..", "yellow"))
				algo.update(batch_campaign_ids, batch_users, batch_clicks, timestamp_raw)
				algo.log_budgets(log_output, total_impressions, timestamp_raw)
				batch_users, batch_clicks, batch_campaign_ids = list(), list(), list()

			impressions_per_campaign[assigned_campaign] += 1
			clicks_per_campaign[assigned_campaign] += simulated_click
			
			# print('.', end='', flush=True)	
			if total_impressions % 100000 == 0 and len(batch_clicks) > 0:
				all_metrics = get_basic_metrics(all_model_impressions, all_model_prediction_values)
				batch_metrics = get_basic_metrics(cur_model_impressions, cur_model_prediction_values)

				cur_model_impressions, cur_model_prediction_values = list(), list()
				print('{} {:.2%} Clicks:{} Imp:{} BatchCTR:{:.3%} CumCTR:{:.3%} CumMSE:{:.03} CumMMSE:{:.03}'.format(
					date,
					cur_file_impressions/meta.total_lines[meta.campaign_id][date], 
					all_metrics["Clicks"],
					all_metrics["Impressions"],
					batch_metrics["CTR"],
					all_metrics["CTR"],
					all_metrics["MSE"],
					all_metrics["MMSE"]))

				output.write("{},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{}\n".format(
					all_metrics["Clicks"], 
					all_metrics["Impressions"], 
					total_impressions, 
					timestamp_raw, 
					batch_metrics["CTR"],
					all_metrics["CTR"],
					batch_metrics["MSE"],
					batch_metrics["MMSE"],
					all_metrics["MSE"],
					all_metrics["MMSE"],
					testMeta.get_algo_column_info()))
				output.flush()
				
		input.close()
		algo.update(batch_campaign_ids, batch_users, batch_clicks, timestamp_raw)
		algo.log_budgets(log_output, total_impressions, timestamp_raw)
		batch_users, batch_clicks, batch_campaign_ids = list(), list(), list()
	
	for campaign_id in campaign_ids:
		output_campaign_log.write("{},{},{},{}\n".format(campaign_id, impressions_per_campaign[campaign_id], clicks_per_campaign[campaign_id], testMeta.get_algo_column_info()))
	output_campaign_log.flush()

output.close()
log_output.close()


















