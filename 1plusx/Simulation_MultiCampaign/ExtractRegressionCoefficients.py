import pandas as pd
import os 
import glob
import math
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv
from termcolor import colored

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
from Algos.TargetSplitType import TargetSplitType 

path = "../../RawData/Campaigns/"
impressions_file_path_extension = "/Processed/sorted_time_impressions.csv"
hindsight_multi_campaign_path = "../../RawData/Campaigns/5/Processed/SimulationHindsight"

campaign_ids = [847460, 866128, 856805, 858140, 865041]#, ,  

output = open("{}/RegressionCoefficients.csv".format(hindsight_multi_campaign_path), "w")
output.write("CampaignId,Coefficients,Intercept\n")

for campaign_id in campaign_ids:

	print (colored("Starting building model for {}..".format(campaign_id), "green"))
	print("Reading impressions for campaign {}..".format(campaign_id))
	meta = Metadata("Regression", campaign_id = campaign_id, initialize_user_embeddings = True)
	campaign_users, campaign_impressions = meta.read_impressions()
	
	print("Setting up regression for all users...")
	algo = Regression(meta)
	testMeta = TestMetadata(meta)
	testMeta.click_percent = 0.0
	algo.setup(testMeta)
	
	print("Fitting only campaign {} impressions...".format(campaign_id))
	algo.update(campaign_users, campaign_impressions)
	
	# print("Fited:{}".format(np.mean(algo.prediction)))

	coef = np.array(algo.model.coef_.T)
	# predictions = coef.dot(algo.user_embeddings.T) 
	# print("Calculated: {}".format(np.mean(predictions) + algo.model.intercept_))
	coef_str = np.array2string(coef, separator=' ')[1:-1].replace('\n', '')
	
	output.write("{0},{1},{2}\n".format(campaign_id, coef_str, algo.model.intercept_))
	output.flush()

output.close()

	





















