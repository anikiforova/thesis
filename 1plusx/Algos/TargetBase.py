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

		self.const_global_target_budgets = dict()
		self.global_target_budgets = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))
		self.global_consumed_budgets = dict(zip(self.campaign_ids, np.zeros(len(self.campaign_ids))))

		self.local_target_budgets = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))
		self.expected_impression_count = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))

		self.normalization_values = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))
		
		self.initialize_daily_impressions()
		self.initialize_impression_approximations()

	def initialize_daily_impressions(self):
		daily_impressions = read_csv("{0}/{1}/Processed/daily_impression_breakdown.csv".format(self.meta.base_path, self.meta.campaign_id), sep=",")
		
		daily_impressions["Date"] = daily_impressions["Timestamp"].apply(lambda a: dt.datetime.fromtimestamp(a/1000))

		daily_impressions = daily_impressions.loc[daily_impressions['Date'] >= self.start_date]
		daily_impressions = daily_impressions.loc[daily_impressions['Date'] <= self.end_date]
		daily_impressions = daily_impressions.groupby(["CampaignId"]).agg({"Impressions":np.sum})
		daily_impressions.reset_index(level=0, inplace=True)
		
		self.const_global_target_budgets = dict(zip(daily_impressions["CampaignId"].values, daily_impressions["Impressions"].values))

	def initialize_impression_approximations(self):
		hourly_agg = read_csv("{0}/{1}/Processed/24_avg_impression_breakdown.csv".format(self.meta.base_path, self.meta.campaign_id), sep=",")

		self.hourly_impression_agg = dict()
		for hour in range(0, 24):
			for campaign_id in self.campaign_ids:
				self.hourly_impression_agg[(hour, campaign_id)] = hourly_agg[(hourly_agg["Hour"] == hour) & (hourly_agg["CampaignId"] == campaign_id)]["Impressions"]
				
		#print(self.hourly_impression_agg)		


	def setup(self, testMeta):

		# dividing by the #campaigns since exploration scavanging will reduce the number
		# Scale total campaign impressions 
		self.testMeta = testMeta
		self.elapsed_days = 0

		if not self.testMeta.target_algo:
			print("Setup of targets - Do nothing..")
			return

		for campaign_id in self.campaign_ids:
			self.global_consumed_budgets[campaign_id] = 0
			self.global_target_budgets[campaign_id] = self.get_normalized_target(self.const_global_target_budgets[campaign_id])	

		self.reset_local_target_budgets()
		
	def consume_campaign_budget(self, campaign_id):
		if not self.testMeta.target_algo:
			return

		self.local_target_budgets[campaign_id] -= 1
		self.global_consumed_budgets[campaign_id] += 1

		if self.local_target_budgets[campaign_id] <= 0:
			if self.local_target_budgets[campaign_id] == 0:
				return True
			self.local_target_budgets[campaign_id] = 0

		return False
	
	def consume_target_budgets(self, consumption_log_by_campaign_ids):
		if not self.testMeta.target_algo:
			return

		for campaign_id in self.campaign_ids:
			consumed_budget = np.sum([1 for id in consumption_log_by_campaign_ids if id == campaign_id])

			self.local_target_budgets[campaign_id] -= consumed_budget
			self.global_consumed_budgets[campaign_id] += consumed_budget

			if self.local_target_budgets[campaign_id] < 0:
				self.local_target_budgets[campaign_id] = 0

	def start_new_day(self):
		self.elapsed_days += 1

	def reset_expected_impression_count(self, timestamp):
		if not self.testMeta.target_algo:
			return

		print("Reseting expected impression count. Updating normalization values")
		hour = dt.datetime.fromtimestamp(timestamp/1000).hour

		for campaign_id in self.campaign_ids:
			self.expected_impression_count[campaign_id] = 0
			for i in range(0, self.testMeta.hours):		
				cur_hour = (hour + i) % 24
				self.expected_impression_count[campaign_id] += self.get_normalized_target(self.hourly_impression_agg[(cur_hour, campaign_id)])

			self.calculate_normalization_value(campaign_id)

	def get_normalized_target(self, target):
		if not self.testMeta.target_algo:
			return

		result = 0
		if self.testMeta.is_simulation: # if is simulation then full set of impressions should be used.
			result = int(self.testMeta.target_percent * target)	
		else:
			result = int((self.testMeta.target_percent * target )/ (1.5 * self.campaign_count))

		return result

	def reset_local_target_budgets(self):
		if not self.testMeta.target_algo:
			return

		print("Updating budgets...")
		print_warning = False
		for campaign_id in self.campaign_ids:
			campaign_remaining_budget = self.global_target_budgets[campaign_id] - self.global_consumed_budgets[campaign_id]
			campaign_remaining_budget = max(0, campaign_remaining_budget)
			if self.testMeta.target_split == TargetSplitType.DAILY:
				self.local_target_budgets[campaign_id] = int(campaign_remaining_budget / (self.total_days - self.elapsed_days))
			else:
				self.local_target_budgets[campaign_id] = campaign_remaining_budget
				print_warning = True
				
			
			self.calculate_normalization_value(campaign_id)

		if print_warning:
			print(colored("WARNING: Using NO_SPLIT target budget type.", "yellow")) 
	
	def calculate_normalization_value(self, campaign_id):
		if not self.testMeta.target_algo:
			return

		if self.testMeta.crop_minimal_target and self.local_target_budgets[campaign_id] < self.testMeta.crop_percent * self.expected_impression_count[campaign_id]:
			print(colored("Crop target", "red"))
			self.local_target_budgets[campaign_id] = 0				

		if self.testMeta.normalize_target_value:
			self.local_target_budgets[campaign_id] /= self.expected_impression_count[campaign_id]

		#print("{} - {}: {} < {} * {}".format(campaign_id, self.testMeta.crop_minimal_target, self.local_target_budgets[campaign_id], self.testMeta.crop_percent, self.expected_impression_count[campaign_id]))
			
		self.normalization_values[campaign_id] = round(((math.log(self.local_target_budgets[campaign_id] + 1) + 1)) ** self.testMeta.target_alpha, 2)

	def log_budgets(self, log_output, timestamp, total_impressions):
		if not self.testMeta.target_algo:
			return

		local_target_budgets_str = ",".join([str(self.local_target_budgets[cid]) for cid in self.campaign_ids])
		global_target_budgets_str = ",".join([str(int(self.global_target_budgets[cid] - self.global_consumed_budgets[cid])) for cid in self.campaign_ids])
		
		algo_column_info = self.testMeta.get_algo_column_info()
		log_output.write("Local,{},{},{},{}\n".format(timestamp, total_impressions, local_target_budgets_str, algo_column_info))
		log_output.write("Global,{},{},{},{}\n".format(timestamp, total_impressions, global_target_budgets_str, algo_column_info))
		log_output.flush()

	def initialize_revenue(self):
		if not self.testMeta.target_algo:
			return

		campaign_revenue_meta = read_csv("{}/CampaignMetadata.csv".format(self.meta.base_path), sep=",")
		campaigns = campaign_revenue_meta["CampaignId"].values
		CPC = campaign_revenue_meta["CPC"].values
		CPM = campaign_revenue_meta["CPM"].values

		self.CPC = dict(zip(campaigns, CPC))
		self.CPM = dict(zip(campaigns, CPM))

	def get_remaining_target_budgets(self):
		return self.local_target_budgets

	def get_normalized_estimate(self, ctr, campaign_id):
		# revenue_estimate = (ctr_estimate * self.meta.CPC[campaign_id] + self.meta.CPM[campaign_id] / 1000) 
		return ctr * self.normalization_values[campaign_id]






