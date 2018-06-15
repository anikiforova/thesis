import numpy as np
import math
import random

from sklearn import linear_model

class EGreedy_Lin:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.learning_rate = 10

		self.articles_users = dict()
		self.articles_clicks = dict()
		self.articles_models = dict()
		self.articles_fit = dict()

	def add_new_article(self, article_id):
		if article_id not in self.articles_clicks.keys():
			self.articles_users[article_id] =  np.array([])
			self.articles_clicks[article_id] = np.array([])
			self.articles_models[article_id] = linear_model.LinearRegression()
			self.articles_fit[article_id] = False

	def update(self, user, selected_article, click):
		
		self.articles_users[selected_article] = np.append(self.articles_users[selected_article], user)
		self.articles_clicks[selected_article] = np.append(self.articles_clicks[selected_article], click)
		
		cur_len = len(self.articles_clicks[selected_article])
		if  cur_len % self.learning_rate == 0 and cur_len > 0:
			self.articles_users[selected_article] = self.articles_users[selected_article].reshape([cur_len, 6 ])
			self.articles_models[selected_article].fit(self.articles_users[selected_article], self.articles_clicks[selected_article])
			self.articles_fit[selected_article] = True

	def warmup(self, fo):

		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha

		articles = list()
		best_value = 0
		selected_article = -1

		same_values = list()
		for line in lines:
			article_id = int(line.split(" ")[0])
			articles.append(article_id)
			
			self.add_new_article(article_id)			
			
			if self.articles_fit[article_id] == False:
				cur_value = 0
			else:
				cur_value = self.articles_models[article_id].predict([user])
			
			if best_value < cur_value:
				best_value = cur_value
				selected_article = article_id
				same_values = list([article_id])

			elif best_value == cur_value:
				same_values.append(article_id)

		selected_article = np.random.choice(same_values, 1)
			
		if explore:
			selected_article = np.random.choice(articles, 1)

		return selected_article, False

