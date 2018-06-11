import numpy as np
from random import randint
import math
from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import norm
import scipy.linalg

class LinUCB_GP_All:
	
	def __init__(self, alpha, cluster_count):
		self.d = 100
		self.beta = 0.05
		self.training_size = 100
		self.warmup_impressions = 0
		self.cluster_count = cluster_count
		self.error = 0.001 # np.random.uniform(0.01, 0.1)

		# number of lines that will be used to be updated together
		# self.batch_size = 100
		# self.error = np.random.uniform(0, 0.01)
		self.articles_to_update = list()
		self.clicks_to_update = list()

		self.clusters = self.get_clusters()
		self.articles = np.array([])
		self.users = np.array([])
		self.clicks = np.array([])

		self.articles_to_update = list()
		self.users_to_update = list()
		self.clicks_to_update = list()

		self.K = np.identity(self.training_size)
		print(self.cluster_count)
		self.k_articles = np.identity(self.cluster_count)

		for i in range(0, self.cluster_count):
			self.k_articles[i][i] = 1 
			for j in range(i+1, self.cluster_count):
				self.k_articles[i][j] = self.exponential_kernel(self.clusters[i], self.clusters[j])
				self.k_articles[j][i] = self.k_articles[i][j]

		self.L = np.identity(self.training_size)
		self.A = np.identity(self.training_size)

	def warmup(self, file):
		pass

	def get_batch_size(self):
		if len(self.articles) < 100:
			return 100
		return len(self.articles) / 10

	def get_clusters(self):
		clusters_input = open("../../1plusx/Clusters.csv", "r")
		cluster_centers = np.array([])
		cluster_count = 0
		for line in clusters_input:
			center = np.fromstring(line[1:-1], sep=" ")
			cluster_centers = np.append(cluster_centers, center)
			cluster_count += 1
		cluster_centers = cluster_centers.reshape([cluster_count, self.d])
		clusters_input.close()
		return cluster_centers

	def select(self, user, pre_selected, click):
		selected_article = -1
		warmup = False 

		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			self.update(user, pre_selected, click)
			selected_article = int(pre_selected)
			warmup = True
		else:	
			ucb = float("-inf")
			for c in range(0, self.cluster_count):
				mu, var = self.get_article_metrics_cholesky(user, c)
				cur_ucb = mu + self.beta * var

				if cur_ucb >= ucb:
					selected_article = c
					ucb = cur_ucb
			# print(str(selected_article) + " " + str(warmup))
		return selected_article, warmup

	def update(self, user, selected_article, click):
		self.articles_to_update.append(selected_article)
		self.users_to_update.append(user)
		self.clicks_to_update.append(click)

		if len(self.articles_to_update) >= self.get_batch_size():
			print("Updating...")

			self.articles = np.append(self.articles, self.articles_to_update)
			articles_count = len(self.articles)
			self.users = np.append(self.users, self.users_to_update).reshape([articles_count, self.d])
			self.clicks = np.append(self.clicks, self.clicks_to_update)
		
			self.K = np.identity(articles_count)

			for i in range(0, articles_count):
				self.K[i][i] = 1 + self.error
				for j in range(i+1, articles_count):
					# print(self.articles[i])
					self.K[i][j] = self.kernel(self.users[i], int(self.articles[i]), self.users[j], int(self.articles[j]))
					self.K[j][i] = self.K[i][j]

			print("Done with kernel calculating. staring cholesky")		
			# print(self.K)
			self.L = scipy.linalg.cholesky(self.K, lower=True)
			b = scipy.linalg.solve_triangular( self.L, self.clicks)
			self.A = scipy.linalg.solve_triangular( self.L.transpose(), b)

			self.articles_to_update = list()
			self.users_to_update = list()
			self.clicks_to_update = list()
			print("Done Updating...")

	def exponential_kernel(self, a, b):
		return np.exp(-0.5*np.sum((a-b)*(a-b))/(self.beta**2))

	def kernel_users(self, user1, user2):
		return self.exponential_kernel(user1, user2)
		# return 1
		
	def kernel_articles(self, article_id1, article_id2):
		result = 1
		try:
			result = self.k_articles[article_id1][int(article_id2)] 
		except IndexError as ex:
			print(str(article_id1) + " " + str(article_id2) + " " + str(ex))
			raise
		return result

	def kernel(self, user1, article_id1, user2, article_id2):
		return self.kernel_users(user1, user2) * self.kernel_articles(article_id1, article_id2)

	def get_article_metrics_cholesky(self, user, article_id):
		user_kernels = np.array(list(map(lambda o: self.kernel_users(user, o), self.users)))
		# user_kernels = 1
		article_kernels =  np.array(list(map(lambda o: self.kernel_articles(article_id, o), self.articles)))
		cur_k = user_kernels * article_kernels	

		mu = cur_k.reshape([1, len(cur_k)]).dot(self.A)
		v = scipy.linalg.solve_triangular( self.L, cur_k)
		var = self.kernel(user, article_id, user, article_id) + v.transpose().dot(v)
		return mu, var
