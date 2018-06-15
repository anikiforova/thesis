import numpy as np
import math
import random

from Util import to_vector 
from sklearn import linear_model

class EGreedy_Lin_Hybrid:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.learning_rate = 100
		self.dimentions = 36	

		self.pairs = np.array([])
		self.clicks = np.array([])
		self.articles = dict()
		self.bad_articles = set()
		# self.model = linear_model.LinearRegression()
		self.model = linear_model.SGDClassifier(loss='hinge', penalty='l2')
		self.is_fitted = False 
		
	def kernel(self, user, article_id):
		return user.dot(self.articles[article_id]).reshape([1, self.dimentions])
		# return np.append(user, self.articles[article_id])
		# return user

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if (article_id not in self.articles) and (article_id not in self.bad_articles):
			try:
				article = to_vector(line)
			except IndexError:
				#	print("Skipping line, weird formatting.." + str(article_id))
				self.bad_articles.add(article_id)
				return -1

			self.articles[article_id] = article.reshape([1, 6])
		
		if article_id in self.bad_articles:
			return -1
		return article_id
			
	def update(self, user, selected_article, click):
		pair = self.kernel(user.reshape([6, 1]), selected_article)
		self.pairs = np.append(self.pairs, pair)
		self.clicks = np.append(self.clicks, click)
		
		cur_len = len(self.clicks)
		if cur_len % self.learning_rate == 0 and cur_len > 0:
			self.pairs = self.pairs.reshape([cur_len, self.dimentions])
			self.model = self.model.partial_fit(self.pairs, self.clicks, [0, 1])
			# self.model.fit(self.pairs, self.clicks)
			self.is_fitted = True

	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha

		cur_articles = list()
		best_value = -1000000
		selected_article = -1

		same_values = list()
		user = user.reshape([6, 1])
		for line in lines:
			article_id = self.add_new_article(line)	
			
			if article_id == -1: continue

			cur_articles.append(article_id)		
			if self.is_fitted: 
				pair = self.kernel(user, article_id)

				cur_value = self.model.predict(pair.reshape(1, self.dimentions))
				
				if best_value < cur_value:
					best_value = cur_value
					selected_article = article_id
					same_values = list([article_id])

				elif best_value == cur_value:
					same_values.append(article_id)

			else:
				same_values.append(article_id)

		selected_article = np.random.choice(same_values, 1)
			
		if explore:
			selected_article = np.random.choice(cur_articles, 1)

		return selected_article, False

