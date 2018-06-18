import numpy as np
import math
import random

class TS:
	
	def __init__(self, alpha):
		self.alpha = alpha

		self.articles_clicks = dict()
		self.sample_size = 100
		#self.bootstraps = dict()

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
		if article_id not in self.articles_mean:
			self.articles_clicks[article_id] = np.array([])
		return article_id

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article] = np.append(self.articles_clicks[selected_article], click)	

	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		best_value_articles = list()
		best_value = 0
		selected_article = -1

		for line in lines:
			article_id = self.add_new_article(line)			
			sample = np.random.choice(self.articles_clicks[article_id], self.sample_size, True)
			cur_value = np.sum(sample) / self.sample_size

			if best_value < cur_value:
				best_value = cur_value
				best_value_articles = list([article_id])
				
			elif best_value == cur_value:
				best_value_articles.append(article_id)
			
		index = random.randint(0, len(best_value_articles)-1)	
		selected_article = best_value_articles[index]

		return selected_article, False

