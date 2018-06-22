import numpy as np
import math
from numpy.linalg import inv
from scipy.optimize import minimize

from Util import to_vector

class TS_RLR:
	
	def __init__(self, alpha):
		self.d = 6
		self.k = 6

		self.alpha = alpha
		self.batch_size = 100
		# self.training_size = 10000
		# self.impressions = 1
		self.batch_ids = list([])
		self.batch_clicks = list([])

		self.articles_1_d = list([])
		self.articles_d_1 = list([])
		self.article_ids = dict()
		self.bad_articles = set()

		self.mu = list([])
		self.q = list([])

	
	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if article_id in self.bad_articles:
			return -1

		if article_id not in self.article_ids:
			try:
				article = to_vector(line)
			except IndexError:
				# print("Skipping line, weird formatting.." + str(article_id))
				self.bad_articles.add(article_id)
				return -1

			self.article_ids[article_id] = len(self.article_ids)
			self.articles_1_d.append(article.reshape([1, self.d]))
			self.articles_d_1.append(article.reshape([self.d, 1]))
			self.mu.append(0)
			self.q.append(self.alpha)

		return article_id

	def to_minimize(w):
		part1 = 1/2 * sum (self.q * (w - self.m) * (self.w - self.m)) 
		self.batch_articles = self.articles_d_1[self.batch_ids]
		part2 = 0
		for article, click in self.batch_articles, self.batch_clicks:
			part2 += math.log(1+math.exp(-click * w.dot(article))) 
		
		return part1 + part2

	def update(self, user, selected_article, click):
		self.impressions += 1
		self.batch_ids.append(self.article_ids[selected_article])
		self.batch_clicks.append(click)

		if self.impressions % self.batch_size == 0:
			w = np.random.normal(self.d)
			res = minimize(self.to_minimize, w, method='Newton-CG', options={'xtol': 1e-8, 'disp': True})
		
	
	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		selected_article = -1
		warmup = False
		
		if self.impressions < self.training_size:
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
				
				list_article_id = self.article_ids[article_id]
				cur_value = np.random.normal(self.mu[list_article_id], 1 / self.q[list_article_id])
		
				if cur_value > value:
					selected_article = article_id
					value = cur_value

		return selected_article, warmup





