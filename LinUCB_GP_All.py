import numpy as np
from random import randint
import math
from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import norm
import scipy.linalg
from Util import to_vector

class LinUCB_GP_All:
	
	def __init__(self, alpha):
		self.d = 6
		self.beta = 0.05
		self.training_size = 100
		self.warmup_impressions = 0

		# number of lines that will be used to be updated together
		self.batch_size = 100

		self.articles_to_update = list()
		self.users_to_update = list()
		self.clicks_to_update = list()

		self.distinct_articles = dict()
		self.bad_articles = set()
		self.k_articles = dict()

		self.articles = np.array([])
		self.users = np.array([])
		self.clicks = np.array([])

		self.K = np.identity(self.training_size)
		
		self.L = np.identity(self.training_size)
		self.A = np.identity(self.training_size)

	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		selected_article = -1
		warmup = False 

		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			for line in lines:
				article_id = self.add_new_article(line)
				if article_id == -1: # invalid article
					continue
			self.update(user, pre_selected_article, click)
			selected_article = pre_selected_article
			warmup = True
		else:	

			ucb = -10000.0
			for line in lines:
				article_id = self.add_new_article(line)
				if article_id == -1: # invalid article
					continue

				mu, var = self.get_article_metrics_cholesky(user, article_id)
				cur_ucb = mu + self.beta * var

				if cur_ucb >= ucb:
					selected_article = article_id
					ucb = cur_ucb
			# print(str(selected_article) + " " + str(warmup))
		return selected_article, warmup

	def update(self, user, selected_article, click):	
		self.articles_to_update.append(selected_article)
		self.users_to_update.append(user)
		self.clicks_to_update.append(click)

		if len(self.articles_to_update) >= self.batch_size:
			print("Updating...")

			self.articles = np.append(self.articles, self.articles_to_update)
			articles_count = len(self.articles)
			self.users = np.append(self.users, self.users_to_update).reshape([articles_count, self.d])
			self.clicks = np.append(self.clicks, self.clicks_to_update)
		
			self.K = np.identity(articles_count)

			for i in range(0, articles_count):
				self.K[i][i] = 1
				for j in range(i+1, articles_count):
					self.K[i][j] = self.kernel(self.users[i], self.articles[i], self.users[j], self.articles[j])
					self.K[j][i] = self.K[i][j]

			print("Done with kernel calculating. staring cholesky")		
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
		return self.k_articles[article_id1][article_id2] 		

	def kernel(self, user1, article_id1, user2, article_id2):
		return self.kernel_users(user1, user2) * self.kernel_articles(article_id1, article_id2)

	def get_article_metrics_cholesky(self, user, article_id):
		user_kernels = 1 #np.array(list(map(lambda o: self.kernel_users(user, o), self.users)))
		article_kernels =  np.array(list(map(lambda o: self.kernel_articles(article_id, o), self.articles)))
		cur_k = user_kernels * article_kernels	

		mu = cur_k.reshape([1, len(cur_k)]).dot(self.A )
		v = scipy.linalg.solve_triangular( self.L, cur_k)
		var = self.kernel(user, article_id, user, article_id) + v.transpose().dot(v)
		return mu, var

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if article_id in self.bad_articles:
			return -1

		if article_id not in self.articles:
			try:
				article = to_vector(line)
			except IndexError:
				print("Skipping line, weird formatting.." + str(article_id))
				self.bad_articles.add(article_id)
				return -1

			self.k_articles[article_id] = dict()
				
			for a in self.distinct_articles.keys():
				value = self.exponential_kernel(article, self.distinct_articles[a])
				self.k_articles[a][article_id] = value
				self.k_articles[article_id][a] = value
				
			self.distinct_articles[article_id] = article
			self.k_articles[article_id][article_id] = 1

		return article_id