import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase

class TS_Lin(AlgoBase):
	
	def __init__(self, meta):
		super(TS_Lin, self).__init__(meta)	
		self.impressions = 100

	def setup(self):
		super(TS_Lin, self).setup()

		self.B = np.identity(self.meta.dimensions)
		self.B_i = np.identity(self.meta.dimensions)
		self.cov = np.identity(self.meta.dimensions)
		
		self.mu = np.zeros(self.meta.dimensions).reshape([1, self.meta.dimensions])
		self.f  = np.zeros(self.meta.dimensions).reshape([1, self.meta.dimensions])

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks)
		train_user_count = len(clicks)

		for user_id, click in zip(users, clicks):
			embedding = self.user_embeddings[user_id]
			self.B += embedding.reshape([self.meta.dimensions, 1]).dot(embedding.reshape([1, self.meta.dimensions]))
			if click:
				self.f += embedding

		self.B_i = inv(self.B)
		self.cov = self.meta.alpha * self.B_i 
		self.mu = list(np.array(self.B_i.dot(self.f.reshape([self.meta.dimensions, 1]))).flat)

		sample_mu = np.random.multivariate_normal(self.mu, self.cov, self.user_count)
		for index, embedding in enumerate(self.user_embeddings):
			self.prediction[index] = embedding.dot(sample_mu[index])
		
		self.impressions += train_user_count
		self.predictionPosprocessing(users, clicks)		
		print(" Done.")



