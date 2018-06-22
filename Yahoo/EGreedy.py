import numpy as np
import math
import random

class EGreedy:
	
	def __init__(self, alpha):
		self.alpha = alpha

		self.articles_clicks = dict()
		self.articles_impressions = dict()
		self.articles_mean = dict()
		self.articles_var = dict()

	def add_new_article(self, article_id):
		if article_id not in self.articles_mean:
			self.articles_clicks[article_id] = 0.0
			self.articles_impressions[article_id] = 0.0
			self.articles_mean[article_id] = 0.0
			self.articles_var[article_id] = 0.0

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article] += click
		self.articles_impressions[selected_article] += 1.0

		p = self.articles_clicks[selected_article] / self.articles_impressions[selected_article]
		q = 1.0 - p

		self.articles_mean[selected_article] = p
		self.articles_var[selected_article] = p*q

	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha
		# print(EGreedy.alpha)
		articles = list()
		best_mean = 0
		selected_article = -1
		best_articles = list()

		if explore:
			for line in lines:
				article_id = int(line.split(" ")[0])
				articles.append(article_id)
				
				self.add_new_article(article_id)	

			selected_article = np.random.choice(articles, 1)

		else:
			for line in lines:
				article_id = int(line.split(" ")[0])
				self.add_new_article(article_id)			
				
				cur_mean = self.articles_mean[article_id]
				if best_mean < cur_mean:
					best_mean = cur_mean
					best_articles = list([article_id])

				elif best_mean == cur_mean:
					best_articles.append(article_id)
			
			selected_article = np.random.choice(best_articles, 1)
			
		return selected_article, False

