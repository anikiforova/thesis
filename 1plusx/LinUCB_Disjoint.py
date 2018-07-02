import numpy as np
import math
from numpy.linalg import inv

from Util import to_vector

class LinUCB_Disjoint:
	
	def __init__(self, alpha, user_embeddings, user_ids):
		self.d = 100
		self.alpha = alpha
		
		self.user_embeddings = user_embeddings 
		self.user_ids = user_ids
		self.user_hast_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))

		self.A = np.identity(self.d)
		self.A_i =  np.identity(self.d)
		self.b =  np.zeros(self.d)

		self.predition = np.ones(self.user_count) * 0.02 + np.random.normal(0, 0.01, self.user_count) # randomize initialization

	def update(self, users, clicks):
		user_ids = self.user_hast_to_id[users]
		embeddings = self.user_embedding[user_ids]
		count_train_users = len(clicks)

		# self.A += sum(embeddings.reshape([self.d, count_train_users]).dot(embeddings.reshape([count_train_users, self.d])))
		# self.A_i = inv(self.A)
		# self.b += sum(self.A.dot(clicks))

		for user_id, click in users, clicks:		
			user = self.user_embedding[self.user_hast_to_id[user_id]]

			self.A += user.reshape([self.d, 1]).dot(user.reshape([1, self.d]))
			self.A_i = inv(self.A)
			if click == 1 :
				self.b += user

		cur_theta = self.A_i.dot(self.b)
		print(cur_theta.shape)
		print(self.user_embedding.shape)
		
		mean = self.user_embedding.dot(cur_theta)
		var = self.user_embedding.reshape([self.user_count, self.d]).dot(self.A_i).dot(self.user_embedding)

		self.predition = mean + self.alpha * np.sqrt(var)

	def get_recommendations(self, count):
		recommendation_ids = self.predition.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return np.array(recommendation_hashes)






