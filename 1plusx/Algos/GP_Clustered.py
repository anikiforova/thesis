import math
import numpy as np
from numpy.linalg import inv

from sklearn.gaussian_process.kernels import Matern, RBF

from .AlgoBase import AlgoBase
from .TestMetadata import TestMetadata

class GP_Clustered(AlgoBase):
	
	def __init__(self, meta):
		print("Starting GP_Clustered setup ...")	
		super(GP_Clustered, self).__init__(meta)
		self.meta.gp_running_algo = True		
		self.meta.kernel_name = "Matern"
		print("Done with Setup.")

	def setup(self, testMetadata):
		super(GP_Clustered, self).setup(testMetadata)
		self.cluster_embeddings = self.testMeta.read_cluster_embeddings()
		self.u_to_c  			= self.testMeta.read_user_assignments()

		kernel = Matern(length_scale = self.testMeta.length_scale, nu=self.testMeta.nu) 
		self.K = kernel(self.cluster_embeddings) + np.identity(self.testMeta.cluster_count) * self.testMeta.noiseVar
		self.K_u_c = kernel(self.user_embeddings, self.cluster_embeddings)
		self.k_u_u = np.array([kernel([embedding]) for embedding in self.user_embeddings])

		self.clicks_per_cluster 	 = np.zeros(self.testMeta.cluster_count) 
		self.impressions_per_cluster = np.ones(self.testMeta.cluster_count) 
		self.ctr 					 = np.zeros(self.testMeta.cluster_count).reshape([self.testMeta.cluster_count, 1])

	def update(self, users, clicks):
		print("Starting Update.. ")
		users, clicks = self.prepareClicks(users, clicks)

		new_impressions = np.zeros(self.testMeta.cluster_count)
		for user_id, click in zip(users, clicks):
			user_cluster_id = self.u_to_c[user_id]
			self.clicks_per_cluster[user_cluster_id] += click
			new_impressions[user_cluster_id] += 1.0
		
		# print("Difference in impression counts:")
		# print(new_impressions)
		self.impressions_per_cluster += new_impressions 
		self.ctr = self.clicks_per_cluster / self.impressions_per_cluster  
		variance_change = self.testMeta.noiseVar / self.impressions_per_cluster
		# print("Variance change:")
		# print(variance_change)

		cur_k = self.K + np.diag(variance_change) 
		cur_k_inv = inv(cur_k)
		# print("Covariance matrix:")
		# print(cur_k_inv)

		print("Done with inversion..")
		for user_index in np.arange(0, self.user_count):
			mid = self.K_u_c[user_index].reshape([1, self.testMeta.cluster_count]).dot(cur_k_inv) # 1 x C
			var = self.k_u_u[user_index] - mid.dot(self.K_u_c[user_index].reshape([self.testMeta.cluster_count, 1]))
			mean = mid.dot(self.ctr) # UxC . Cx1 = Ux1
			
			# self.prediction[user_index] = np.random.normal(mean, np.sqrt(var))
			self.prediction[user_index] = mean + self.testMeta.alpha * np.sqrt(var)
			#diff += (self.prediction[user_index] - new_value)**2
			#self.prediction[user_index] = new_value
			if user_index < 5:
				print("mean:{0} stdev:{1} value:{2} ".format(mean, np.sqrt(var), self.prediction[user_index]))


		print("Done with prediction.")
		super(GP_Clustered, self).predictionPosprocessing(users, clicks)		
		print(" Done.")






