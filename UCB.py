import numpy as np
import math
import random

from Util import to_vector

class UCB:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.training_size = 10000

		self.articles_mean = dict()
		self.articles_var = dict()
		self.articles_clicks = dict()

	def add_new_article(self, article_id):
		if article_id not in self.articles_mean:
			self.articles_clicks[article_id] = list()
			self.articles_mean[article_id] = 0
			self.articles_var[article_id] = 0

	def update(self, user, selected_article, click):
		self.articles_clicks[selected_article].append(click)
		self.articles_mean[selected_article] = np.mean(self.articles_clicks[selected_article])
		self.articles_var[selected_article] = np.var(self.articles_clicks[selected_article])
		
	def warmup(self, file):
		total_impressions = 0
		for line in file:
			if total_impressions > self.training_size:
				break

			total_impressions += 1
			line = line.split("|")
			no_space_line = line[0].split(" ")
			pre_selected_article = int(no_space_line[1])
			click = int(no_space_line[2])
			user = to_vector(line[1])

			self.add_new_article(pre_selected_article)
			self.update(user, pre_selected_article, click)

	def select(self, user, lines, total_impressions):
		best_mean = 0
		selected_article = -1

		for line in lines:
			article_id = int(line.split(" ")[0])
			self.add_new_article(article_id)
			cur_mean = self.articles_mean[article_id]

			if best_mean < cur_mean:
				best_mean = cur_mean
				selected_article = article_id

		return selected_article

