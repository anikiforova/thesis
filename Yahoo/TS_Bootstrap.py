import numpy as np
import math
import random

class TS_Bootstrap:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.training_size = 1000
		self.warmup_impressions = 0
		self.bootstrap_sample = 1000

		self.articles_clicks = dict()
		self.articles_fail = dict()

	def add_new_article(self, article_id):
		if article_id not in self.articles_clicks:
			self.articles_clicks[article_id] = list()
		return article_id

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article].append(click)
		
	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		best_value_articles = list()
		best_value = 0
		selected_article = -1
		warmup = False
		
		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			self.add_new_article(pre_selected_article)
			self.update(user, pre_selected_article, click)
			selected_article = pre_selected_article
			warmup = True
		else:
			for line in lines:
				article_id = int(line.split(" ")[0])
				self.add_new_article(article_id)			

				clicks = sum(self.articles_clicks[article_id])
				impressions = len(self.articles_clicks[article_id])
				if  impressions < self.bootstrap_sample:
					cur_value = np.random.beta(clicks + 1, impressions - clicks + 1)
				else:
					sample = np.random.choice(self.articles_clicks[article_id], self.bootstrap_sample, True)	
					cur_value = sum(sample) / len(sample)

				if best_value < cur_value:
					best_value = cur_value
					best_value_articles = list([article_id])

				elif best_value == cur_value:
					best_value_articles.append(article_id)
			
			index = random.randint(0, len(best_value_articles)-1)	
			selected_article = best_value_articles[index]

		return selected_article, warmup

