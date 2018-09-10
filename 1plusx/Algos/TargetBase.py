import numpy as np
import math
import datetime as dt
from pandas import read_csv
from pandas import DataFrame
from termcolor import colored

from TargetSplitType import TargetSplitType

class TargetBase:
	
	def __init__(self, meta, campaign_ids, start_date, end_date):
		self.meta = meta
		self.testMeta = ""
		self.campaign_ids = campaign_ids
		self.campaign_count = len(self.campaign_ids)
		self.total_days = (end_date - start_date).days + 1 # number of days to process
		self.start_date = start_date
		self.end_date = end_date

		self.total_impressions = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))
		self.target_budgets = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))
		self.normalization_values = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))
		self.consumed_budget = dict(zip(self.campaign_ids, np.zeros(len(self.campaign_ids))))
		self.initialize_daily_impressions()

	def initialize_daily_impressions(self):
		daily_impressions = read_csv("{0}/{1}/Processed/daily_impression_breakdown.csv".format(self.meta.base_path, self.meta.campaign_id), sep=",")
		
		daily_impressions["Date"] = daily_impressions["Timestamp"].apply(lambda a: dt.datetime.fromtimestamp(a/1000))

		daily_impressions = daily_impressions.loc[daily_impressions['Date'] >= self.start_date]
		daily_impressions = daily_impressions.loc[daily_impressions['Date'] <= self.end_date]
		daily_impressions = daily_impressions.groupby(["CampaignId"]).agg({"Impressions":np.sum})
		daily_impressions.reset_index(level=0, inplace=True)
		
		self.absolute_total_impressions = dict(zip(daily_impressions["CampaignId"].values, daily_impressions["Impressions"].values))

	def setup(self, testMeta):
		# dividing by the #campaigns since exploration scavanging will reduce the number
		# Scale total campaign impressions 
		self.testMeta = testMeta
		self.elapsed_days = 0

		for campaign_id in self.total_impressions.keys():
			self.consumed_budget[campaign_id] = 0
			self.total_impressions[campaign_id] = int((self.testMeta.target_percent * self.absolute_total_impressions[campaign_id] )/(2*self.campaign_count))
			self.target_budgets[campaign_id] = self.total_impressions[campaign_id]
		
	def calculate_normalization_value(self, target):
		return round(((math.log(target + 1) + 1)) ** self.testMeta.target_alpha, 2)

	def update_target_budgets(self, consumed_budget):
		for campaign_id in self.campaign_ids:
			self.target_budgets[campaign_id] -= consumed_budget[campaign_id]
			self.consumed_budget[campaign_id] += consumed_budget[campaign_id]

			if self.target_budgets[campaign_id] < 0:
				self.target_budgets[campaign_id] = 0

			# maybe the remainder of impressions has to be considered as a parameter
			self.normalization_values[campaign_id] = self.calculate_normalization_value(self.target_budgets[campaign_id])

	def start_new_day(self):
		self.elapsed_days += 1

	def recalculate_budgets(self):
		print("Updating budgets...")
		for campaign_id in self.campaign_ids:
			campaign_remaining_budget = self.total_impressions[campaign_id] - self.consumed_budget[campaign_id]

			if self.testMeta.target_split == TargetSplitType.NO_SPLIT:
				self.target_budgets[campaign_id] = campaign_remaining_budget

			elif self.testMeta.target_split == TargetSplitType.DAILY:
				self.target_budgets[campaign_id] = int(campaign_remaining_budget / (self.total_days - self.elapsed_days))
			
			else:
				print(colored("WARNING: No target budget split type defined.", "yellow")) 
				self.target_budgets[campaign_id] = campaign_remaining_budget

			if self.target_budgets[campaign_id] < 0:
				self.target_budgets[campaign_id] = 0
			self.normalization_values[campaign_id] = self.calculate_normalization_value(self.target_budgets[campaign_id])
		
	def log_budgets(self, log_output, timestamp, total_impressions):
		daily_target_budgets_str = ",".join([str(self.target_budgets[cid]) for cid in self.campaign_ids])
		total_target_budgets_str = ",".join([str(int(self.total_impressions[cid] - self.consumed_budget[cid])) for cid in self.campaign_ids])
		
		algo_column_info = self.testMeta.get_algo_column_info()
		log_output.write("Daily,{},{},{},{}\n".format(timestamp, total_impressions, daily_target_budgets_str, algo_column_info))
		log_output.write("Total,{},{},{},{}\n".format(timestamp, total_impressions, total_target_budgets_str, algo_column_info))
		log_output.flush()

	def initialize_revenue(self):
		campaign_revenue_meta = read_csv("{}/CampaignMetadata.csv".format(self.meta.base_path), sep=",")
		campaigns = campaign_revenue_meta["CampaignId"].values
		CPC = campaign_revenue_meta["CPC"].values
		CPM = campaign_revenue_meta["CPM"].values

		self.CPC = dict(zip(campaigns, CPC))
		self.CPM = dict(zip(campaigns, CPM))

	def get_remaining_target_budgets(self):
		return self.target_budgets

	def get_normalized_estimate(self, ctr, campaign_id):
		# revenue_estimate = (ctr_estimate * self.meta.CPC[campaign_id] + self.meta.CPM[campaign_id] / 1000) 
		return ctr * self.normalization_values[campaign_id]






