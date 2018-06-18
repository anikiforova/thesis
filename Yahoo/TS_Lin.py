import numpy as np
import math
from numpy.linalg import inv

from Util import to_vector

class TS_Lin:
	
	def __init__(self, alpha):
		# self.d = 6
		self.d = 36
		self.alpha = alpha
		self.trained_impressions = 0
		self.training_size = 10000
		self.warmup_impressions = 0

		self.B = np.identity(self.d)
		self.B_i = np.identity(self.d)
		self.cov = np.identity(self.d)
		self.R = 0.1
		self.v_2 = 1 - alpha #(self.R **2) * 24 * self.d / self.alpha # constant used to scale the covariance matrix. these params have to be tweaked 

		self.mu = np.zeros(self.d).reshape([1, self.d])
		self.f = np.zeros(self.d).reshape([1, self.d])

		self.articles_1_d = dict()
		self.articles_d_1 = dict()
		self.bad_articles = set()

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if article_id in self.bad_articles:
			return -1

		if article_id not in self.articles_1_d:
			try:
				article = to_vector(line)
			except IndexError:
				# print("Skipping line, weird formatting.." + str(article_id))
				self.bad_articles.add(article_id)
				return -1

			# self.articles_1_d[article_id] = article.reshape([1, self.d])
			# self.articles_d_1[article_id] = article.reshape([self.d, 1])
			self.articles_1_d[article_id] = article.reshape([1, 6])
			self.articles_d_1[article_id] = article.reshape([6, 1])

		return article_id

	def update(self, user, selected_article, click):
		self.trained_impressions += 1
		pair = np.outer(user, self.articles_1_d[selected_article]).reshape([1, 36])

		# self.B = self.B + self.articles_d_1[selected_article].dot(self.articles_1_d[selected_article])
		self.B = self.B + pair.reshape([self.d, 1]).dot(pair.reshape([1, self.d]))
		if click:
			# self.f = self.f + self.articles_1_d[selected_article]
			self.f = self.f + pair

		if self.trained_impressions % 100 == 0:
			self.B_i = inv(self.B)
			self.cov = self.v_2 * self.B_i 

		self.mu = list(np.array(self.B_i.dot(self.f.reshape([self.d, 1]))).flat)
	
	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		selected_article = -1
		warmup = False

		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			for line in lines:
				self.add_new_article(line)

			self.update(user, pre_selected_article, click)
			selected_article = pre_selected_article
			warmup = True
		
		else:
			value = 0.0
			sample_mu = np.random.multivariate_normal(self.mu, self.cov)

			for line in lines:
				article_id = self.add_new_article(line)
				if article_id == -1 : 
					continue
				
				pair = np.outer(user, self.articles_1_d[article_id]).reshape([1, 36])
				cur_value = pair.dot(sample_mu)
				# cur_value = self.articles_1_d[article_id].dot(sample_mu)

				if cur_value > value:
					selected_article = article_id
					value = cur_value

		return selected_article, warmup





