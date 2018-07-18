import numpy as np
import math
from numpy.linalg import inv
from random import randint

class Random:
	
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, click_percent = 0.5, equalize_clicks = False, filter_clickers = False, soft_click = False):
		#np.random.seed(seed=3432)
		self.user_ids = user_ids
		self.user_hash_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))
		
		self.user_count = len(self.user_hash_to_id)
		self.prediction = np.ones(self.user_count) * 0.02
		
		self.indexes = np.arange(0, len(self.user_ids)-1)

	def setup(self, alpha):
		pass

	def warmup(self, input, warmup_impression_count):
		pass

	def update(self, users, clicks):
		# self.prediction = np.random.uniform(0, 1, len(self.user_ids))
		self.prediction = np.random.randint(0, 2, len(self.user_ids))
		
	def getPrediction(self, user):
		return self.prediction[self.user_hash_to_id[user]]

	def get_recommendations(self, count):
		recommendation_ids = self.prediction.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)




