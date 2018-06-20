import numpy as np
import math
import random

class TS_Truncated:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.size = 10000
		self.index = 0

		self.articles_clicks = dict()
		self.articles_indexes = dict()
		self.articles_size = dict()

	def add_new_article(self, article_id):
		if article_id not in self.articles_clicks:
			self.articles_clicks[article_id] = np.zeros(self.size)
			self.articles_size[article_id] = 0.0
			self.articles_indexes[article_id] = 0

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article][self.articles_indexes[selected_article]] = click
		self.articles_indexes[selected_article] = (self.articles_indexes[selected_article] + 1) % self.size
		if self.articles_size[selected_article] < self.size:
			self.articles_size[selected_article] = self.articles_size[selected_article] + 1

	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		best_value_articles = list()
		best_value = -1000
		selected_article = -1

		for line in lines:
			article_id = int(line.split(" ")[0])
			self.add_new_article(article_id)			
			
			# cur_value = np.random.normal(self.articles_mean[article_id], self.articles_var[article_id])
			pos = np.sum(self.articles_clicks[article_id])
			neg = self.articles_size[article_id] - pos
			# print(pos + " " + neg)
			cur_value = np.random.beta(pos + 1, neg + 1, 1)
			if best_value < cur_value:
				best_value = cur_value
				best_value_articles = list([article_id])
			elif best_value == cur_value:
				best_value_articles.append(article_id)
			
		index = random.randint(0, len(best_value_articles)-1)	
		selected_article = best_value_articles[index]

		return selected_article, False

