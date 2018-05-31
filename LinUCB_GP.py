import numpy as np
import random
import math
from numpy.linalg import inv
from numpy.linalg import det
from scipy.stats import norm
from Util import to_vector

class LinUCB_GP:
	
	def __init__(self, alpha):
		self.d = 6
		self.beta = 0.05
		self.training_size = 10000
		self.warmup_impressions = 0
		self.selection_size = 100

		# number of lines that will be used to be updated together
		self.batch_size = 500

		self.articles_to_update = list()
		self.users_to_update = list()
		self.clicks_to_update = list()

		self.articles = dict()
		self.bad_articles = set()
		self.k_articles = dict()

		self.selected_articles = np.array([])
		self.selected_users = np.array([])
		self.selected_clicks = np.array([])

		self.K = np.identity(self.selection_size)
		self.K_i = np.identity(self.selection_size)

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
			ucb = -10.0
			for line in lines:
				article_id = self.add_new_article(line)
				if article_id == -1: # invalid article
					continue

				mu, var = self.get_article_metrics(user, article_id, self.K_i)
				cur_ucb = mu + self.beta * var

				if cur_ucb >= ucb:
					selected_article = article_id
					ucb = cur_ucb
			# print(str(selected_article) + " " + str(warmup))
		return selected_article, warmup

	def update(self, user, selected_article, click):
		if len(self.selected_articles) < self.selection_size:
			# initially populate the kernel before reaching the selection size
			cur_count = len(self.selected_articles)
			# print(self.selected_articles)
			
			self.selected_articles = np.append(self.selected_articles, selected_article)
			self.selected_users = np.append(self.selected_users, user).reshape([cur_count+1, self.d])
			self.selected_clicks = np.append(self.selected_clicks, click)
			# print(selected_article)
			# print(self.selected_articles)

			for i in range(0, cur_count):
				self.K[i][cur_count] = self.kernel(self.selected_users[i], self.selected_articles[i], user, selected_article)
				self.K[cur_count][i] = self.K[i][cur_count]

			self.K[cur_count][cur_count] = 1

			if cur_count + 1 == self.selection_size:
				self.K_i = inv(self.K)
				# print(self.K)
		else:
			# print(self.K)
			self.articles_to_update.append(selected_article)
			self.users_to_update.append(user)
			self.clicks_to_update.append(click)
			self.update_batch()
		
	def update_batch(self):
		if len(self.articles_to_update) >= self.batch_size:
			# print("Updating batch...")
			max_value = 0
			worst_i = -1
			best_j = -1
			best_K = self.K

			for i in range(0, self.selection_size):
				for j in range(0, self.batch_size):
					article_id = self.selected_articles[i]
					user = self.selected_users[i]
					click = self.selected_clicks[i]
					self.selected_articles[i] = self.articles_to_update[j]
					self.selected_users[i] = self.users_to_update[j]
					self.selected_clicks[i] = self.clicks_to_update[j]	

					value, new_K = self.calculate_cur_value()
					if value > max_value:
						max_value = value
						worst_i = i
						best_j = j
						best_K = new_K

					self.selected_articles[i] = article_id
					self.selected_users[i] = user
					self.selected_clicks[i] = click
			
			# print("\nBest " + str(best_j) + " worst " + str(worst_i))
			self.selected_articles[worst_i] = self.articles_to_update[best_j]
			self.selected_users[worst_i] = self.users_to_update[best_j]
			self.selected_clicks[worst_i] = self.clicks_to_update[best_j]
			self.K = best_K
			self.K_i = inv(self.K)

			self.articles_to_update = list()
			self.users_to_update = list()
			self.clicks_to_update = list()

	def exponential_kernel(self, a, b):
		return np.exp(-0.5*np.sum((a-b)*(a-b))/(self.d**2))

	def kernel_users(self, user1, user2):
		return self.exponential_kernel(user1, user2)
		# return 1
		
	def kernel_articles(self, article_id1, article_id2):
		return self.k_articles[article_id1][article_id2] 		

	def kernel(self, user1, article_id1, user2, article_id2):
		return self.kernel_users(user1, user2) * self.kernel_articles(article_id1, article_id2)

	def calculate_cur_value(self, i):
		# print('.', end='', flush=True)
		cov = self.K.copy()	
		for j in range(0, self.selection_size):
			cov[i][j] = self.kernel(self.selected_users[i], self.selected_articles[i], self.selected_users[j], self.selected_articles[j])
			cov[j][i] = cov[i][j]
			if cov[i][j] == 1 and i != j:
				return 0, self.K

		cov_i = inv(cov)
		results = 0
		for i in range(0, self.selection_size):
			mu, var = self.get_article_metrics(self.selected_users[i], self.selected_articles[i], cov_i)
			results += mu #norm(mu, var).pdf(self.selected_clicks[i])

		return results, cov

	def get_article_metrics(self, user, article_id, cov_i):
		user_kernels = 1
		article_kernels =  np.array(list(map(lambda o: self.kernel_articles(article_id, o), self.selected_articles)))

		cur_k = user_kernels * article_kernels		
		pre_cur_k_K_i = cur_k.dot(cov_i)

		mu = pre_cur_k_K_i.dot(self.selected_clicks) 
		var = self.kernel(user, article_id, user, article_id) - pre_cur_k_K_i.dot(cur_k.reshape(self.selection_size, 1))
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
				
			for a in self.articles.keys():
				value = self.exponential_kernel(article, self.articles[a])
				self.k_articles[a][article_id] = value
				self.k_articles[article_id][a] = value
				
			self.articles[article_id] = article
			self.k_articles[article_id][article_id] = 1

		return article_id