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

	def update(self, batch_campaign_ids, users, clicks, timestamp):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks) # this is not great, filtering could be applied that would mess up ordering.
		train_user_count = len(clicks)

		for index, campaign_id in enumerate(batch_campaign_ids):
			embedding = self.user_embeddings[users[index]]
			self.A[campaign_id] += embedding.reshape([self.meta.dimensions, 1]).dot(embedding.reshape([1, self.meta.dimensions]))
			if clicks[index] == 1 :
				self.b[campaign_id] += embedding

		for campaign_id in self.campaign_ids:
			self.A_i[campaign_id] = inv(self.A[campaign_id])
			self.theta[campaign_id] = self.A_i[campaign_id].dot(self.b[campaign_id]) # [self.d, self.d] x [self.d, 1] = [self.d, 1]

		self.reset_expected_impression_count(timestamp)
		self.update_multi_campaign_predictions(self.campaign_ids)	

	def update_single_prediction(self, embedding, campaign_id):
		mean = max(0.00000001, embedding.dot(self.theta[campaign_id]))
		var = math.sqrt(embedding.reshape([1, self.meta.dimensions]).dot(self.A_i[campaign_id]).dot(embedding))
				
		ctr_estimate = mean + self.testMeta.alpha * var		
		normalized_estimate = ctr_estimate

		if self.testMeta.normalize_ctr:
			normalized_estimate = self.get_normalized_estimate(ctr_estimate, campaign_id)
		
		return ctr_estimate, normalized_estimate
	
	# 1) select users based on normalized prediction value (by the campaign they come from)
	# 2) select top percent from each campaign and make that the recommendation list	
	def get_recommendations(self, percent):
		recommendation_ids = np.array([])
		prediction_argsorted = self.prediction.argsort()

		for campaign_id in self.campaign_ids:
			prediction_assignment = self.campaign_assignment == campaign_id
			assigned_user_count = np.sum(prediction_assignment)

			selected_user_count = int(assigned_user_count * percent)
			recommendation_ids = np.append(recommendation_ids, prediction_argsorted[prediction_assignment][-selected_user_count:][::-1])

		
		expected_user_count = int(self.user_count * percent)
		print("Expected user count: {}, actual: {}".format(expected_user_count, len(recommendation_ids)))

		recommendation_hashes = set([ self.user_id_to_hash[x] for x in recommendation_ids ])

		return recommendation_hashes

	def get_recommendations_normalized(self, percent):
		recommendation_ids = np.array([])
		self.normalized_predictions = self.prediction
	
		for campaign_id in self.campaign_ids:
			prediction_assignment = self.campaign_assignment == campaign_id
			self.normalized_predictions[prediction_assignment] = np.mean(self.predictions[prediction_assignment])

		recommendation_ids = self.normalized_predictions.argsort()[-count:][::-1]
		recommendation_hashes = set([ self.user_id_to_hash[x] for x in recommendation_ids ])

		return recommendation_hashes














