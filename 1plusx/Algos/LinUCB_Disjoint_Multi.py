import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase
from TargetBase import TargetBase
from Metadata import Metadata

class LinUCB_Disjoint_Multi(AlgoBase, TargetBase):
	
	def __init__(self, meta, campaign_ids, start_date, end_date):
		AlgoBase.__init__(self, meta)
		TargetBase.__init__(self, meta, campaign_ids, start_date, end_date)

	def setup(self, testMeta):
		AlgoBase.setup(self, testMeta)
		TargetBase.setup(self, testMeta)

		self.A 		= dict()
		self.A_i	= dict()
		self.b 		= dict()
		self.theta  = dict()

		for campaign_id in self.campaign_ids:
			self.A[campaign_id] 	= np.identity(self.meta.dimensions)
			self.A_i[campaign_id]	= np.identity(self.meta.dimensions)
			self.b[campaign_id] 	= np.zeros(self.meta.dimensions)
			self.theta[campaign_id] = np.zeros(self.meta.dimensions)

	def update(self, batch_campaign_ids, users, clicks, discard_target_impressions):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks) # this is not great, filtering could be applied that would mess up ordering.
		train_user_count = len(clicks)

		consumed_budget = dict(zip(self.campaign_ids, np.zeros(len(self.campaign_ids))))
		for index, campaign_id in enumerate(batch_campaign_ids):
			consumed_budget[campaign_id] += 1
			embedding = self.user_embeddings[users[index]]
			self.A[campaign_id] += embedding.reshape([self.meta.dimensions, 1]).dot(embedding.reshape([1, self.meta.dimensions]))
			if clicks[index] == 1 :
				self.b[campaign_id] += embedding

		for campaign_id in self.campaign_ids:
			self.A_i[campaign_id] = inv(self.A[campaign_id])
			self.theta[campaign_id] = self.A_i[campaign_id].dot(self.b[campaign_id]) # [self.d, self.d] x [self.d, 1] = [self.d, 1]

		if not discard_target_impressions:
			self.update_target_budgets(consumed_budget)	
		
		self.update_multi_campaign_predictions()
		
		if not discard_target_impressions:
			self.recalculate_budgets()

	def update_single_prediction(self, embedding, campaign_id):
		mean = max(0.00000001, embedding.dot(self.theta[campaign_id]))
		var = math.sqrt(embedding.reshape([1, self.meta.dimensions]).dot(self.A_i[campaign_id]).dot(embedding))
				
		ctr_estimate = mean + self.testMeta.alpha * var		
		normalized_estimate = self.get_normalized_estimate(ctr_estimate, campaign_id)
		return ctr_estimate, normalized_estimate

			













