import numpy as np
import math
import random

# from Util import to_vector

class UCB:
	
	def __init__(self, alpha, clusters, training_size_scale = 1):
		self.alpha = alpha
		self.clusters = clusters
		self.training_size = 10000 * training_size_scale
		self.warmup_impressions = 0

		self.articles_clicks = np.zeros(clusters)
		self.articles_impressions = np.zeros(clusters)
		self.articles_mean = np.zeros(clusters)
		self.articles_var = np.zeros(clusters)

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article] += click
		self.articles_impressions[selected_article] += 1.0

		p = self.articles_clicks[selected_article] / self.articles_impressions[selected_article]
		q = 1.0 - p

		self.articles_mean[selected_article] = p
		self.articles_var[selected_article] = p*q
		
	def select(self, user, pre_selected, click):
		selected_article = -1
		warmup = False

		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			self.update(user, pre_selected, click)
			selected_article = pre_selected
			warmup = True
		else:
			selected_article = np.argmax(list(map(lambda i: self.articles_mean[i] + self.alpha * self.articles_var[i], range(0, self.clusters))))
		
		return selected_article, warmup

