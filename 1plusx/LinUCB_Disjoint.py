import numpy as np
import math
from numpy.linalg import inv

class LinUCB_Disjoint:
	
	
	def __init__(self, alpha, user_embeddings, user_ids):
		self.d = 100
		self.alpha = alpha
		
		# [len(user_ids), self.d]
		self.user_embeddings = user_embeddings   
		self.user_embeddings_t = np.transpose(user_embeddings)
		self.user_ids = user_ids
		self.user_hast_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))
		self.user_count = len(self.user_ids)

		self.A = np.identity(self.d)
		self.A_i =  np.identity(self.d)
		self.b =  np.zeros(self.d)

		# random prediction before first update
		self.predition = np.ones(self.user_count) * 0.02 + np.random.normal(0, 0.01, self.user_count) 

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		train_user_count = len(clicks)

		embeddings = np.array([ self.user_embeddings[self.user_hast_to_id[x]] for x in users ]).reshape([train_user_count, self.d])
		embeddings_t = np.transpose(embeddings) 
		
		self.A += sum(embeddings_t.dot(embeddings)) # [self.d, len(users)] x [len(users), self.d] = [self.d, self.d]
		self.A_i = inv(self.A)
		self.b += sum(embeddings_t.dot(clicks)) # [self.d, len(users)] x [len(users), 1] = [self.d, 1]
		
		cur_theta = self.A_i.dot(self.b) # [self.d, self.d] x [self.d, 1] = [self.d, 1]
		mean = self.user_embeddings.dot(cur_theta).reshape([self.user_count, 1]) # [len(users), self.d] x [self.d, 1] = [len(users), 1]
		var = np.apply_along_axis(lambda user: user.dot(self.A_i).dot(user.reshape(self.d, 1)), 1, self.user_embeddings)
		
		self.predition = mean + self.alpha * np.sqrt(var) # this should be std dev but need to normalize users first
		self.predition = np.array([item for sublist in self.predition for item in sublist])
		print(" Done.")

	def get_recommendations(self, count):
		recommendation_ids = self.predition.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)






