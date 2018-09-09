import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase
from TargetBase import TargetBase
from Metadata import Metadata

class EGreedy_Multi(AlgoBase, TargetBase):
	
	def __init__(self, meta, campaign_ids, start_date, end_date):
		AlgoBase.__init__(self, meta)
		TargetBase.__init__(self, meta, campaign_ids, start_date, end_date)

	def setup(self, testMeta):
		AlgoBase.setup(self, testMeta)
		TargetBase.setup(self, testMeta)

		self.clicks = dict(zip(self.campaign_ids, np.zeros(len(self.campaign_ids))))
		self.impressions = dict(zip(self.campaign_ids, np.ones(len(self.campaign_ids))))

	def update(self, batch_campaign_ids, users, clicks, discard_target_impressions):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks) # this is not great, filtering could be applied that would mess up ordering.
		train_user_count = len(clicks)

		consumed_budget = dict(zip(self.campaign_ids, np.zeros(len(self.campaign_ids))))
		for index, campaign_id in enumerate(batch_campaign_ids):
			consumed_budget[campaign_id] += 1
			self.clicks[campaign_id] += clicks[index]
			self.impressions[campaign_id] += 1

		if not discard_target_impressions:
			self.update_target_budgets(consumed_budget)	
		
		self.update_multi_campaign_predictions()
		
		if not discard_target_impressions:
			self.recalculate_budgets()

		print("\nCTR: ", end='', flush=True)
		for campaign_id in self.campaign_ids:
			print("{}:{:.04} ".format(campaign_id, self.clicks[campaign_id]/ self.impressions[campaign_id]), end='', flush=True)
		print("")

	def update_single_prediction(self, embedding, campaign_id):
		ctr_estimate = self.clicks[campaign_id] / self.impressions[campaign_id]
		#q = 1.0 - p
		#mean = p 
	#	var = p * q
		
		#ctr_estimate = mean + self.testMeta.alpha * math.sqrt(var)		
		normalized_estimate = self.get_normalized_estimate(ctr_estimate, campaign_id)
		return ctr_estimate, normalized_estimate

			













