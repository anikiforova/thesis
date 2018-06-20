import numpy as np
import math
import random

class TS_Gibbs:
	
	def __init__(self, alpha):
		self.alpha = alpha

		self.articles_clicks = dict()
		
	def add_new_article(self, article_id):
		if article_id not in self.articles_clicks:
			self.articles_clicks[article_id] = list()

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article].append(click)		

	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		best_value_articles = list()
		best_value = -1
		selected_article = -1

		for line in lines:
			article_id = int(line.split(" ")[0])
			self.add_new_article(article_id)			
			
			if len(self.articles_clicks[article_id]) == 0:
				cur_value = 0.5
			else:
				index = random.randint(0, len(self.articles_clicks[article_id])-1)	
				cur_value = self.articles_clicks[article_id][index]

			if best_value < cur_value:
				best_value = cur_value
				best_value_articles = list([article_id])
			elif best_value == cur_value:
				best_value_articles.append(article_id)
			
		index = random.randint(0, len(best_value_articles)-1)	
		selected_article = best_value_articles[index]

		return selected_article, False

