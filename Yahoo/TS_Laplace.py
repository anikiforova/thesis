import numpy as np
import math
import random

class TS_Laplace:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.training_size = 1000
		self.warmup_impressions = 0

		self.articles_success = dict()
		self.articles_fail = dict()

	def add_new_article(self, article_id):
		if article_id not in self.articles_success:
			self.articles_success[article_id] = 0.0
			self.articles_fail[article_id] = 0.0

		return article_id

	def update(self, user, selected_article, click):
		self.articles_success[selected_article] += click
		self.articles_fail[selected_article] += 1 - click

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

				total_tries = self.articles_success[article_id] + self.articles_fail[article_id]	
				if total_tries == 0:
					cur_value = np.random.beta(1, 1)
				else:
					a = self.articles_success[article_id] + 1e-6 - 1
					b = self.articles_fail[article_id] + 1e-6 - 1
					mode = a / ( a + b)
					hessian = a / mode + b / (1-mode)
					cur_value = mode + np.sqrt(1 / hessian) * np.random.randn(1)
					
				if best_value < cur_value:
					best_value = cur_value
					best_value_articles = list([article_id])

				elif best_value == cur_value:
					best_value_articles.append(article_id)
			
			index = random.randint(0, len(best_value_articles)-1)	
			selected_article = best_value_articles[index]

		return selected_article, warmup

