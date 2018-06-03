import numpy as np
import math
import random
from random import randint

class EGreedy:
	
	def __init__(self, alpha, clusters):
		self.alpha = alpha
		self.clusters = clusters
		
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
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha

		if explore:
			selected_article = randint(0, self.clusters)
		else:
			# print(np.argmax(user))
			selected_article = np.argmax(user)

		return selected_article, not explore

