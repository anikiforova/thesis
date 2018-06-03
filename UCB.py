import numpy as np
import math
import random

from Util import to_vector

class UCB:
	
	def __init__(self, alpha, training_size_scale = 1):
		self.alpha = alpha
		self.training_size = 10000 * training_size_scale
		self.warmup_impressions = 0

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
		
	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		selected_article = -1
		warmup = False
		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			self.add_new_article(pre_selected_article)
			self.update(user, pre_selected_article, click)
			selected_article = pre_selected_article
			warmup = True
		else:
			best_ucb = 0
			for line in lines:
				article_id = int(line.split(" ")[0])
				self.add_new_article(article_id)
				cur_mean = self.articles_mean[article_id]
				cur_var = self.articles_var[article_id]

				if best_ucb < cur_mean + self.alpha * cur_var:
					best_ucb = cur_mean + self.alpha * cur_var
					selected_article = article_id

		return selected_article, warmup

