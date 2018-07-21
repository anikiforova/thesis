import numpy as np
import math
from numpy.linalg import inv


from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from AlgoBase import AlgoBase

class GP_Clustered(AlgoBase):
	
	def __init__(self, meta):

		print("Starting GP_Clustered setup ...")	
		super(GP_Clustered, self).__init__(meta)
			
		self.cluster_embeddings = self.meta.read_cluster_embeddings()
		self.u_to_c  			= self.meta.read_user_assignments()

		self.K = np.identity(self.meta.cluster_count)
		self.calculate_kernel()
		self.K += np.identity(self.meta.cluster_count) * self.meta.noiseVar
		
		self.K_u_c = np.zeros(self.user_count * self.meta.cluster_count).reshape([self.user_count, self.meta.cluster_count])
		self.k_u_u = np.zeros(self.user_count)

		if self.meta.calculate_users_kernels:
			self.calculate_users_to_clusters()
			self.meta.save_user_clusters(self.K_u_c)
		else:
			self.K_u_c = self.meta.read_user_clusters(self.user_count)

		self.calculate_user_kernels();	

		print("Done with Setup.")

	def setup(self):
		super(GP_Clustered, self).setup()

		self.clicks_per_cluster 	 = np.zeros(self.meta.cluster_count) 
		self.impressions_per_cluster = np.ones(self.meta.cluster_count) 
		self.ctr 					 = np.zeros(self.meta.cluster_count).reshape([self.meta.cluster_count, 1])
	
	def kernel(self, x, y):
		return np.exp(-0.5 * np.sum((x-y)*(x-y))/(self.meta.dimensions**2))

	def calculate_kernel(self):
		for x in np.arange(0, self.meta.cluster_count):
			for y in np.arange(0, self.meta.cluster_count):
				self.K[x][y] = self.kernel(self.cluster_embeddings[x], self.cluster_embeddings[y]) 

	def calculate_users_to_clusters(self):
		print("Starting kernels for users calculations...", end='', flush=True)	
		for user_index in np.arange(0, self.user_count):
			self.K_u_c[user_index] = np.apply_along_axis(lambda c: self.kernel(self.user_embeddings[user_index], c), 1, self.cluster_embeddings)
			self.k_u_u[user_index] = self.kernel(self.user_embeddings[user_index], self.user_embeddings[user_index]) 
		print("Done.")

	def calculate_user_kernels(self):
		print("Starting user kernels calculations...", end='', flush=True)	
		for user_index in np.arange(0, self.user_count):
			self.k_u_u[user_index] = self.kernel(self.user_embeddings[user_index], self.user_embeddings[user_index]) 
		print("Done.")

	def update(self, users, clicks):
		print("Starting Update.. ")
		users, clicks = self.prepareClicks(users, clicks)

		new_impressions = np.zeros(self.meta.cluster_count)
		for user_id, click in zip(users, clicks):
			user_cluster_id = self.u_to_c[user_id]
			self.clicks_per_cluster[user_cluster_id] += click
			new_impressions[user_cluster_id] += 1.0
		
		print("Difference in impression counts:")
		print(new_impressions)
		self.impressions_per_cluster += new_impressions 
		self.ctr = self.clicks_per_cluster / self.impressions_per_cluster  
		variance_change = self.meta.noiseVar / self.impressions_per_cluster
		print("Variance change:")
		print(variance_change)

		cur_k = self.K + np.diag(variance_change) 
		cur_k_inv = inv(cur_k)
		# print("Covariance matrix:")
		# print(cur_k_inv)

		print("Done with inversion..")
		diff = 0.0
		for user_index in np.arange(0, self.user_count):
			mid = self.K_u_c[user_index].reshape([1, self.meta.cluster_count]).dot(cur_k_inv) # 1 x C
			var = self.k_u_u[user_index] - mid.dot(self.K_u_c[user_index].reshape([self.meta.cluster_count, 1]))
			mean = mid.dot(self.ctr) # UxC . Cx1 = Ux1
			
			# self.prediction[user_index] = np.random.normal(mean, self.meta.alpha * np.sqrt(var))
			self.prediction[user_index] = mean + self.meta.alpha * np.sqrt(var)
			#diff += (self.prediction[user_index] - new_value)**2
			#self.prediction[user_index] = new_value
			# if user_index < 10:
			# 	print("mean:{0} var:{1} value:{2} ".format(mean, np.sqrt(var), new_value))


		print("Done with prediction.")
		super(GP_Clustered, self).predictionPosprocessing(users, clicks)		
		print(" Done.")






