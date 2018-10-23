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

from Algos.Metadata import Metadata 
from Algos.TestMetadata import TestMetadata 

class SimulationController:

	hindsight_multi_campaign_path = "../../RawData/Campaigns/5/Processed/SimulationHindsight"
	calibration_parameters_filename = "CalibrationParameters.csv"

	chisquare_dfs = [0, 1, 5, 10, 20, 100]
	simulation_indexes = [2, 5, 5, 5, 5, 5] 
	chi_alphas = [2, 4]

	def __init__(self, reinitialize_calibration = False):
		input = open("{}/RegressionCoefficients.csv".format(self.hindsight_multi_campaign_path), "r")
		input.readline()

		self.random_values_count = 100000
		self.random_values = np.random.uniform(0, 1, self.random_values_count)
		self.used_random_values = 0

		self.chi_squares_count = 100000
		self.chi_squares_used = self.chi_squares_count
		self.chi_squares = list()

		self.campaign_ids				= list()
		self.coeff 						= dict()
		self.intercepts					= dict()
		self.calibration 				= dict()
		self.ctr 						= dict()
		self.stdev 						= dict()

		for line in input:
			line_breakdown = line.split(",")

			campaign_id = int(line_breakdown[0])
			self.campaign_ids.append(campaign_id)
			coefficients = np.array(np.fromstring(line_breakdown[1], sep=" "))
			self.coeff[campaign_id] = coefficients
			self.intercepts[campaign_id] = float(line_breakdown[2])

		if reinitialize_calibration:
			self.reinitialize_calibration_parameters()
		else:
			self.initialize_calibration_parameters()


	def setup(self, testMeta):
		self.reset_chi_squares(testMeta.chi_df)
		self.cur_test_meta = testMeta
		
		for campaign_id in self.campaign_ids:
			# print("{},{},{},{}".format(campaign_id, testMeta.chi_alpha, testMeta.chi_df, testMeta.simulation_index))
			if (campaign_id, testMeta.chi_alpha, testMeta.chi_df, testMeta.simulation_index) not in self.calibration.keys():
				self.reinitialize_campaign_calibration_parameters(testMeta, campaign_id)

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

	def calculate_simulated_value(self, simulation_index, chisquare_df, prediction, stdev, ctr, chi_alpha = 1):
		simulated_value = 0.0
		if simulation_index == 0:
			simulated_value = prediction + np.random.uniform(-ctr/10, +ctr/10, 1)
		elif simulation_index == 1:
			simulated_value = np.random.normal(prediction, stdev, 1)
		elif simulation_index == 2:
			simulated_value = np.random.normal(prediction, stdev, 1) + np.random.uniform(-ctr/10, +ctr/10, 1)
		elif simulation_index == 3:
			simulated_value = SimulationController.sigmoid(prediction) + np.random.uniform(-ctr/10, +ctr/10, 1)
		elif simulation_index == 4:
			simulated_value = SimulationController.sigmoid(SimulationController.square_func(prediction)) + np.random.uniform(-ctr/10, +ctr/10, 1)
		elif simulation_index == 5:
			self.chi_squares_used += 1
			if self.chi_squares_used >= self.chi_squares_count:
				self.reset_chi_squares(chisquare_df)

			simulated_value = np.random.normal(prediction, stdev, 1) + chi_alpha * self.chi_squares[self.chi_squares_used] * ctr + np.random.uniform(-ctr/10, +ctr/10, 1)
	
		return simulated_value

	def reset_chi_squares(self, df):
		if df > 0:
			self.chi_squares_used = 0
			self.chi_squares = np.random.noncentral_chisquare(df, 0, self.chi_squares_count)
			self.chi_squares = (self.chi_squares - np.min(self.chi_squares)) / (np.max(self.chi_squares) - np.min(self.chi_squares))

	def initialize_calibration_parameters(self):
		print("Initializing calibration parameters..")
		dataframe = read_csv("{}/{}".format(self.hindsight_multi_campaign_path, self.calibration_parameters_filename), ",")

		groups = dataframe.groupby(["CampaignId", "ChiAlpha", "DegreesOfFreedom", "SimulationIndex"])
		for group in groups.groups.keys(): 
			campaign_id = group[0]
			row = dataframe.loc[(dataframe.CampaignId == campaign_id) & 
								(dataframe.ChiAlpha == group[1]) &
								(dataframe.DegreesOfFreedom == group[2]) & 
								(dataframe.SimulationIndex == group[3])]

			self.ctr[campaign_id] = row.CTR.values[0]
			self.stdev[campaign_id] = row.STDEV.values[0]
			self.calibration[group] = row.Calibration.values[0]
				
		# print(self.calibration)		
		
	def reinitialize_campaign_calibration_parameters(self, testMeta, campaign_id):
		print("Reinitialize calibration parameters for {}..".format(campaign_id))
		output = open("{}/{}".format(self.hindsight_multi_campaign_path, self.calibration_parameters_filename), "a")

		meta = Metadata("Regression", campaign_id = campaign_id, initialize_user_embeddings = True)
		_, campaign_impressions = meta.read_impressions()
		_, user_embeddings = meta.read_user_embeddings()

		predictions = self.coeff[campaign_id].dot(user_embeddings.T) + self.intercepts[campaign_id]

		self.ctr[campaign_id] = np.mean(campaign_impressions)
		self.stdev[campaign_id] = np.std(predictions)
		print("Prediction mean:{}".format(np.mean(predictions)))

		simulation_index = testMeta.simulation_index
		df = testMeta.chi_df
		chi_alpha = testMeta.chi_alpha

		self.reset_chi_squares(df)

		simulated_predictions = np.array([self.calculate_simulated_value(simulation_index, df, p, self.stdev[campaign_id], self.ctr[campaign_id], chi_alpha) for p in predictions ])
			
		simulated_prediction_ctr = np.mean(simulated_predictions) 
		self.calibration[(campaign_id, chi_alpha, df, simulation_index)] = self.ctr[campaign_id] / simulated_prediction_ctr	
		
		output.write("{},{},{},{},{},{},{}\n".format(campaign_id, chi_alpha, df, simulation_index, self.ctr[campaign_id], self.stdev[campaign_id], self.calibration[(campaign_id, chi_alpha, df, simulation_index)]))
		output.flush()

		print(" Done with campaign:{}".format(campaign_id))
		output.close()


	def reinitialize_calibration_parameters(self):
		print("Reinitialize calibration parameters..")

		output = open("{}/{}".format(self.hindsight_multi_campaign_path, self.calibration_parameters_filename), "a")
#		output.write("CampaignId,ChiAlpha,DegreesOfFreedom,SimulationIndex,CTR,STDEV,Calibration\n")

		for campaign_id in self.campaign_ids:
			print("Starting {}.".format(campaign_id))
			meta = Metadata("Regression", campaign_id = campaign_id, initialize_user_embeddings = True)
			_, campaign_impressions = meta.read_impressions()
			_, user_embeddings = meta.read_user_embeddings()

			predictions = self.coeff[campaign_id].dot(user_embeddings.T) + self.intercepts[campaign_id]

			self.ctr[campaign_id] = np.mean(campaign_impressions)
			self.stdev[campaign_id] = np.std(predictions)
			print("Prediction mean:{}".format(np.mean(predictions)))

			for simulation_index, df in zip(self.simulation_indexes, self.chisquare_dfs):
				print("{}.".format(df), end='', flush=True)

				self.reset_chi_squares(df)

				for chi_alpha in self.chi_alphas:
					simulated_predictions = np.array([self.calculate_simulated_value(simulation_index, df, p, self.stdev[campaign_id], self.ctr[campaign_id], chi_alpha) for p in predictions ])
				
					simulated_prediction_ctr = np.mean(simulated_predictions) 
					self.calibration[(campaign_id, df)] = self.ctr[campaign_id] / simulated_prediction_ctr	
					output.write("{},{},{},{},{},{},{}\n".format(campaign_id, chi_alpha, df, simulation_index, self.ctr[campaign_id], self.stdev[campaign_id], self.calibration[(campaign_id, df)]))
					output.flush()

			print(" Done with campaign:{}".format(campaign_id))
		output.close()


	def predict_value(self, campaign_id, user_embedding):
		return self.coeff[campaign_id].dot(user_embedding.reshape([100, 1])) + self.intercepts[campaign_id]

	def get_simulated_impression(self, campaign_id, user_embedding):
		predicted_value = self.predict_value(campaign_id, user_embedding)

		simulated_value = self.calculate_simulated_value(self.cur_test_meta.simulation_index, self.cur_test_meta.chi_df, predicted_value, self.stdev[campaign_id], self.ctr[campaign_id], self.cur_test_meta.chi_alpha)

		group = (campaign_id, self.cur_test_meta.chi_alpha, self.cur_test_meta.chi_df, self.cur_test_meta.simulation_index);
		calibrated_simulated_value = simulated_value * self.calibration[group]

		self.used_random_values += 1
		if self.random_values_count <= self.used_random_values:
			self.random_values = np.random.uniform(0, 1, self.random_values_count)
			self.used_random_values = 0

		impression = 0
		if calibrated_simulated_value >= self.random_values[self.used_random_values]:
			impression = 1

		return impression


