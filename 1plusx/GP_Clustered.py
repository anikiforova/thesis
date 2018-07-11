import numpy as np
import math
from numpy.linalg import inv

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from AlgoBase import AlgoBase

class GP_Clustered(AlgoBase):
	
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, filter_clickers = False, soft_click = False):
		print("Starting GP_Clustered setup ...", end='', flush=True)	
		super(GP_Clustered, self).__init__(user_embeddings, user_ids, filter_clickers, soft_click)

		self.cluster_embeddings = cluster_embeddings
		self.cluster_count = len(self.cluster_embeddings)
		self.d = dimensions
		self.noiseVar = 0.1
		self.K = np.zeros(self.cluster_count * self.cluster_count).reshape([self.cluster_count, self.cluster_count])

		self.calculate_kernel()
		self.K += np.random.normal(0, 0.001, self.cluster_count * self.cluster_count).reshape([self.cluster_count, self.cluster_count]) # add error

		self.K_u_c = np.zeros(self.user_count * self.cluster_count).reshape([self.user_count, self.cluster_count])
		self.u_to_c = np.zeros(self.user_count, dtype=int)
		self.k_u_u = np.zeros(self.user_count)
		self.calculate_users_to_clusters()
		
		# self.kernel = RBF()
		# print(self.kernel())
		print("Done.")

	def setup(self, alpha):
		super(GP_Clustered, self).setup(alpha)

		self.clicks_per_cluster = np.zeros(self.cluster_count) 
		self.impressions_per_cluster = np.ones(self.cluster_count) 
		self.ctr = np.zeros(self.cluster_count).reshape([self.cluster_count, 1])
	
	def kernel(self, x, y):
		return np.exp(-0.5*np.sum((x-y)*(x-y))/(self.d**2))

	def calculate_kernel(self):
		for x in np.arange(0, self.cluster_count):
			for y in np.arange(0, self.cluster_count):
				self.K[x][y] = self.kernel(self.cluster_embeddings[x], self.cluster_embeddings[y]) 

	def calculate_users_to_clusters(self):
		for user_index in np.arange(0, self.user_count):
			self.K_u_c[user_index] = np.apply_along_axis(lambda c: self.kernel(self.user_embeddings[user_index], c), 1, self.cluster_embeddings)
			self.u_to_c[user_index] = int(np.argmax(self.K_u_c[user_index])) 
			self.k_u_u[user_index] = self.kernel(self.user_embeddings[user_index], self.user_embeddings[user_index]) 
			
	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(GP_Clustered, self).prepareClicks(users, clicks)
		
		for user_id, click in zip(users, clicks):
			user_cluster_id = self.u_to_c[user_id]
			self.clicks_per_cluster[user_cluster_id] += click
			self.impressions_per_cluster[user_cluster_id] += 1.0
			self.ctr[user_cluster_id] = self.clicks_per_cluster[user_cluster_id] / self.impressions_per_cluster[user_cluster_id]

		print("Done with updating clicks..")
		# cur_k = self.K + np.diag(1.0/self.impressions_per_cluster) * self.noiseVar
		# cur_k_inv = inv(cur_k)
		cur_k = self.K 
		cur_k_inv = inv(self.K)

		print("Done with inversion..")
		for user_index in np.arange(0, self.user_count):
			mid = self.K_u_c[user_index].reshape([1, self.cluster_count]).dot(cur_k_inv) # 1 x C
			var = self.k_u_u[user_index] - mid.dot(self.K_u_c[user_index].reshape([self.cluster_count, 1]))
			mean = mid.dot(self.ctr) # UxC . Cx1 = Ux1
			
			self.predition[user_index] = mean + self.alpha * np.sqrt(var)
		print("Done with predition..")
		super(GP_Clustered, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

	def get_recommendations(self, count):
		recommendation_ids = self.predition.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)