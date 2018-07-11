import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase

class LinUCB_Disjoint(AlgoBase):
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, filter_clickers = False, soft_click = False):
		super(LinUCB_Disjoint, self).__init__(user_embeddings, user_ids, filter_clickers, soft_click)
		self.d = dimensions

	def setup(self, alpha):
		super(LinUCB_Disjoint, self).setup(alpha)
		self.A = np.identity(self.d)
		self.A_i =  np.identity(self.d)
		self.b =  np.zeros(self.d)
			
	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(LinUCB_Disjoint, self).prepareClicks(users, clicks)
		train_user_count = len(clicks)

		for user_id, click in zip(users, clicks):
			embedding = self.user_embeddings[user_id]
			self.A += embedding.reshape([self.d, 1]).dot(embedding.reshape([1, self.d]))
			if click == 1 :
				self.b += embedding

		self.A_i = inv(self.A)
		theta = self.A_i.dot(self.b) # [self.d, self.d] x [self.d, 1] = [self.d, 1]

		index = 0
		for embedding in self.user_embeddings:
			self.predition[index] = embedding.dot(theta) + self.alpha * math.sqrt(embedding.reshape([1, self.d]).dot(self.A_i).dot(embedding))
			index += 1

		super(LinUCB_Disjoint, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

	def get_recommendations(self, count):
		recommendation_ids = self.predition.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)
