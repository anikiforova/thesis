import numpy as np
import math
import random
from numpy.linalg import inv
from scipy.optimize import minimize

from Util import to_vector

class TS_RLR:
	
	def __init__(self, alpha):
		self.d = 6
		self.k = 6

		self.alpha = alpha
		self.batch_size = 1000
		self.training_size = 1000
		self.impressions = 0
		self.batch_ids = list([])
		self.batch_clicks = np.array([])

		self.articles_1_d = np.array([])
		self.article_ids = dict()
		self.bad_articles = set()

		self.mu = np.zeros(self.d)
		self.q = self.alpha * np.ones(self.d)
		
	def sigmoid(self, x):
		# print(x)
		return 1.0 / (1.0 + math.exp(-x))

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if article_id in self.bad_articles:
			return -1

		if article_id not in self.article_ids:
			try:
				article = to_vector(line)
			except IndexError:
				self.bad_articles.add(article_id)
				return -1
			
			self.article_ids[article_id] = len(self.article_ids)
			self.articles_1_d = np.append(self.articles_1_d, article).reshape([len(self.article_ids), self.d])
			
		return article_id

	def to_minimize(self, w):
		return 1/2 * sum (self.q * (w - self.mu) * (w - self.mu)) + sum(np.log(1+np.exp(-self.batch_clicks * w.dot(self.batch_articles))))
	

	def update(self, user, selected_article, click):
		self.impressions += 1
		self.batch_ids.append(self.article_ids[selected_article])
		self.batch_clicks = np.append(self.batch_clicks, click)

		if self.impressions % self.batch_size == 0:
			w = np.random.normal(0, 1, self.d)
			self.batch_articles = self.articles_1_d[self.batch_ids].reshape([self.d, self.batch_size])

			res = minimize(self.to_minimize, w, method='nelder-mead', options={'xtol': 1e-8, 'disp': False})
			self.m = res.x
			
			p = 1/(1 + np.exp(- self.m.dot(self.batch_articles)))

			for i in np.arange(0, self.d):
					self.q[i] += sum(self.batch_articles[i] * self.batch_articles[i] * p[i] * (1-p[i]))
				
			self.batch_ids = list([])
			self.batch_clicks = np.array([])

	
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
			best_value = 0
			best_value_articles = list()

			sample_w = np.random.multivariate_normal(self.mu, np.diag(1/self.q))
			for line in lines:
				article_id = self.add_new_article(line)
				if article_id == -1 : 
					continue
				
				a_id = self.article_ids[article_id]
				article = self.articles_1_d[a_id]
				
				cur_value = self.sigmoid(sample_w.dot(article))
		
				if cur_value > best_value:
					best_value_articles = list([article_id])
					best_value = cur_value
				elif cur_value == best_value:
					best_value_articles.append(article_id)

			index = random.randint(0, len(best_value_articles)-1)	
			selected_article = best_value_articles[index]

		return selected_article, warmup





