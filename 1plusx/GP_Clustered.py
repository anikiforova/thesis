import numpy as np
import math
from numpy.linalg import inv

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from AlgoBase import AlgoBase

class GP_Clustered(AlgoBase):
	
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, click_percent = 0.5, equalize_clicks = False, filter_clickers = False, soft_click = False):
		print("Starting GP_Clustered setup ...", end='', flush=True)	
		super(GP_Clustered, self).__init__(user_embeddings, user_ids, click_percent, equalize_clicks, filter_clickers, soft_click)
			
		self.cluster_embeddings = cluster_embeddings
		self.cluster_count = len(self.cluster_embeddings)
		self.d = dimensions
		self.noiseVar = 0.1
		# mkernel = RBF()
		# self.K = mkernel(self.cluster_embeddings)
		self.K = np.identity(self.cluster_count)
		self.calculate_kernel()
		self.K += np.identity(self.cluster_count) * self.noiseVar
		
		self.K_u_c = np.zeros(self.user_count * self.cluster_count).reshape([self.user_count, self.cluster_count])
		self.u_to_c = np.zeros(self.user_count, dtype=int)
		self.k_u_u = np.zeros(self.user_count)
		self.calculate_users_to_clusters()
		
		# self.kernel = RBF()
		self.Kernel = RBF()

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
		print("Adding variance:")
		variance_change = self.noiseVar/self.impressions_per_cluster
		print(variance_change)
		cur_k = self.K + np.diag(variance_change) 
		cur_k_inv = inv(cur_k)
		# cur_k = self.K 
		# cur_k_inv = inv(self.K)

		print("Done with inversion..")
		diff = 0.0
		for user_index in np.arange(0, self.user_count):
			mid = self.K_u_c[user_index].reshape([1, self.cluster_count]).dot(cur_k_inv) # 1 x C
			var = self.k_u_u[user_index] - mid.dot(self.K_u_c[user_index].reshape([self.cluster_count, 1]))
			mean = mid.dot(self.ctr) # UxC . Cx1 = Ux1
			
			# /self.prediction[user_index] = np.random.normal(mean, np.sqrt(var))
			new_value = mean + self.alpha * np.sqrt(var)
			diff += (self.prediction[user_index] - new_value)**2
			self.prediction[user_index] = new_value
			if user_index < 10:
				print("mean:{0} var:{1} value:{2} ".format(mean, np.sqrt(var), new_value))


		print("Done with prediction..Difference in predictions {0}".format(diff/self.user_count))
		super(GP_Clustered, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

	def get_recommendations(self, count):
		recommendation_ids = self.prediction.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)