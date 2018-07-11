import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase

class TS_Lin(AlgoBase):
	
	def __init__(self, user_embeddings, user_ids, dimensions, filter_clickers = False, soft_click = False):
		super(TS_Lin, self).__init__(user_embeddings, user_ids, filter_clickers, soft_click)
		self.d = dimensions
		self.impressions = 100

	def setup(self, alpha):
		super(Regression, self).setup(alpha)
		self.B = np.identity(self.d)
		self.B_i = np.identity(self.d)
		self.cov = np.identity(self.d)
		self.R = 0.1
		
		self.mu = np.zeros(self.d).reshape([1, self.d])
		self.f = np.zeros(self.d).reshape([1, self.d])

	def get_v_2(self):
		return self.alpha# ((self.R **2) * 24 * self.d / math.log(self.impressions, 2))*math.log(1/(1-self.alpha), 2)

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(TS_Lin, self).prepareClicks(users, clicks)
		train_user_count = len(clicks)

		for user_id, click in zip(users, clicks):
			embedding = self.user_embeddings[user_id]
			self.B += embedding.reshape([self.d, 1]).dot(embedding.reshape([1, self.d]))
			if click:
				self.f += embedding

		self.B_i = inv(self.B)
		self.cov = self.get_v_2() * self.B_i 
		self.mu = list(np.array(self.B_i.dot(self.f.reshape([self.d, 1]))).flat)

		index = 0
		sample_mu = np.random.multivariate_normal(self.mu, self.cov, self.user_count)
		for embedding in self.user_embeddings:
			self.predition[index] = embedding.dot(sample_mu[index])
			index += 1	

		self.impressions += train_user_count
		super(TS_Lin, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

	def get_recommendations(self, count):
		recommendation_ids = self.predition.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)



