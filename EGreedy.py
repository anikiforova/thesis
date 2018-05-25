import numpy as np
import math
import random

class EGreedy:
	
	def __init__(self, alpha):
		self.alpha = alpha

		self.articles_mean = dict()
		self.articles_clicks = dict()

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article].append(click)
		self.articles_mean[selected_article] = np.mean(self.articles_clicks[selected_article])
		
	def warmup(self, fo):
		pass
		
	def select(self, user, lines, total_impressions):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha

		articles = list()
		best_mean = 0
		selected_article = -1

		for line in lines:
			article_id = int(line.split(" ")[0])
			articles.append(article_id)

			if article_id in self.articles_clicks.keys():	
				cur_mean = self.articles_mean[article_id]
				if best_mean < cur_mean:
					best_mean = cur_mean
					selected_article = article_id
			else:
				self.articles_clicks[article_id] = list()
				self.articles_mean[article_id] = 0

		if explore:
			selected_article = np.random.choice(articles, 1)

		return selected_article

