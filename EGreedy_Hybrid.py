import numpy as np
import math
from numpy.linalg import inv

from Util import to_vector

class EGreedy_Hybrid
	
	def __init__(self, alpha):
		self.d = 6
		self.alpha = alpha
		
		self.A0 = np.identity(self.k)
		self.b0 = np.zeros(self.k)
		self.A0_i = np.identity(self.k)
		self.beta = self.A0_i.dot(self.b0)

		self.A = dict()
		self.A_i = dict()
		self.B = dict()
		self.b = dict()
		self.articles = dict()
		self.bad_articles = set()

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if article_id in self.bad_articles:
			return -1

		if article_id not in self.A:
			try:
				article = to_vector(line)
			except IndexError:
				print("Skipping line, weird formatting.." + str(article_id))
				self.bad_articles.add(article_id)
				return -1

			self.articles[article_id] = article
			self.A[article_id] = np.identity(self.d)
			self.A_i[article_id] = np.identity(self.d)
			self.B[article_id] = np.zeros(self.d * self.k).reshape([self.d, self.k])
			self.b[article_id] = np.zeros(self.d)

	def update(self, user, selected_article, click):
		article = self.articles[selected_article]
		
		z = np.outer(user, article)
		z_1_k = z.reshape([1, self.k])
		z_k_1 = z.reshape([self.k, 1])
		user_1_d = user.reshape([1, self.d])
		user_d_1 = user.reshape([self.d, 1])
		
		self.A[selected_article] += user_d_1.dot(user_1_d) 
		self.B[selected_article] += user_d_1.dot(z_1_k)
		self.A0 += z_k_1.dot(z_1_k)

		self.A_i[selected_article] = inv(self.A[selected_article])
		self.A0_i = inv(self.A0)

		if click == 1:
			self.b[selected_article] += user
			self.b0 += z.reshape([self.k])
	
	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha
		
		selected_article = -1
		articles = list()
		
		if explore:
			for line in lines:
				article_id = int(line.split(" ")[0])
				self.add_new_article(article_id)
			selected_article = np.random.choice(articles, 1)

		else:
			limit = 0.0
			for line in lines:
				article_id = self.add_new_article(line)
				if article_id == -1: # invalid article
					continue

				z = np.outer(user, self.articles[article_id]).reshape([1, self.k])
				cur_theta = self.A_i[article_id].dot((self.b[article_id] - self.B[article_id].dot(self.beta))) 	
				cur_limit = z.dot(self.beta) + user.dot(cur_theta)

				if(cur_limit >= limit):
					selected_article = article_id
					limit = cur_limit

		return selected_article, False





