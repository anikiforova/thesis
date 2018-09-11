import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase
from TargetBase import TargetBase
from Metadata import Metadata

class TS_Lin_Multi(AlgoBase, TargetBase):
	
	def __init__(self, meta, campaign_ids, start_date, end_date):
		AlgoBase.__init__(self, meta)
		TargetBase.__init__(self, meta, campaign_ids, start_date, end_date)

	def setup(self, testMeta):
		AlgoBase.setup(self, testMeta)
		TargetBase.setup(self, testMeta)
		
		self.A 		= dict()
		self.A_i	= dict()
		self.b 		= dict()

		self.cov  = dict()
		self.mu  = dict()

		self.sample_mu  = dict()
		self.sample_index  = dict()
		
		for campaign_id in self.campaign_ids:
			self.A[campaign_id] 	= np.identity(self.meta.dimensions)
			self.A_i[campaign_id]	= np.identity(self.meta.dimensions)
			self.b[campaign_id] 	= np.zeros(self.meta.dimensions)
			
			self.cov[campaign_id] = np.identity(self.meta.dimensions)
			self.mu[campaign_id] = np.zeros(self.meta.dimensions).reshape([1, self.meta.dimensions])

			
	def update(self, batch_campaign_ids, users, clicks, discard_target_impressions):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks) # this is not great, filtering could be applied that would mess up ordering.
		consumed_budget = dict(zip(self.campaign_ids, np.zeros(len(self.campaign_ids))))
		for index, campaign_id in enumerate(batch_campaign_ids):
			consumed_budget[campaign_id] += 1
			embedding = self.user_embeddings[users[index]]
			self.A[campaign_id] += embedding.reshape([self.meta.dimensions, 1]).dot(embedding.reshape([1, self.meta.dimensions]))
			if clicks[index] == 1 :
				self.b[campaign_id] += embedding

		for campaign_id in self.campaign_ids:
			self.A_i[campaign_id] = inv(self.A[campaign_id])
			self.cov[campaign_id] = self.testMeta.alpha * self.A_i[campaign_id]
			self.mu[campaign_id] = list(np.array(self.A_i[campaign_id].dot(self.b[campaign_id].reshape([self.meta.dimensions, 1]))).flat)

			self.sample_mu[campaign_id] = np.random.multivariate_normal(self.mu[campaign_id], self.cov[campaign_id], self.user_count)
			self.sample_index[campaign_id] = 0

		if not discard_target_impressions:
			self.update_target_budgets(consumed_budget)	
		
		self.update_multi_campaign_predictions()
		
		if not discard_target_impressions:
			self.recalculate_budgets()

	def update_single_prediction(self, embedding, campaign_id):
				
		sample_mu = self.sample_mu[campaign_id][self.sample_index[campaign_id]]
		self.sample_index[campaign_id] += 1

		ctr_estimate = sample_mu.dot(embedding.reshape([self.meta.dimensions, 1]))
		normalized_estimate = ctr_estimate

		if self.testMeta.normalize_ctr:
			normalized_estimate = self.get_normalized_estimate(ctr_estimate, campaign_id)
		
		return ctr_estimate, normalized_estimate








