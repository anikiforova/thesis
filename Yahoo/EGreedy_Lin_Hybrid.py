
import numpy as np
import math
import random

from sklearn import linear_model

class EGreedy_Lin_Hybrid:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.learning_rate = 100
		# self.training_size = 10000
		# self.warmup_impressions = 0

		self.a_users = dict()
		self.a_clicks = dict()
		self.a_models = dict()
		self.a_fit = dict()

		self.all_users = np.array([])
		self.all_clicks = np.array([])
		self.model = linear_model.LinearRegression()
		self.fit = False

	def add_new_article(self, article_id):
		if article_id not in self.a_clicks.keys():
			self.a_users[article_id] =  np.array([])
			self.a_clicks[article_id] = np.array([])
			self.a_models[article_id] = linear_model.LinearRegression()
			self.a_fit[article_id] = False

	def update(self, user, selected_article, click):
		self.a_users[selected_article] = np.append(self.a_users[selected_article], user)
		self.a_clicks[selected_article] = np.append(self.a_clicks[selected_article], click)

		self.all_users = np.append(self.all_users, user)
		self.all_clicks = np.append(self.all_clicks, click)
		
		cur_len = len(self.a_clicks[selected_article])
		if  cur_len % self.learning_rate == 0 and cur_len > 0:
			self.a_users[selected_article] = self.a_users[selected_article].reshape([cur_len, 6 ])
			self.a_models[selected_article].fit(self.a_users[selected_article], self.a_clicks[selected_article])
			self.a_fit[selected_article] = True

		total_len = len(self.all_clicks)
		if total_len % self.learning_rate == 0 and total_len > 0:
			self.all_users = self.all_users.reshape([total_len, 6 ])
			self.model.fit(self.all_users, self.all_clicks)
			self.fit = True			

	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		bucket = random.uniform(0, 1)
		warmup = False
		explore = bucket <= self.alpha

		articles = list()
		best_value = -1000
		selected_article = -1

		# if len(self.all_users) < self.training_size:
		# 	self.warmup_impressions += 1
		# 	self.add_new_article(pre_selected_article)
		# 	self.update(user, pre_selected_article, click)
		# 	selected_article = pre_selected_article
		# 	warmup = True
		if explore:
			selected_article = np.random.choice(articles, 1)
		else:
			same_values = list()
			for line in lines:
				article_id = int(line.split(" ")[0])
				articles.append(article_id)
				
				self.add_new_article(article_id)			
				
				cur_value_by_overall_model = 0
				if self.fit:
					cur_value_by_overall_model = self.model.predict([user])
				
				# necessary in case the article_id model is not trained yet
				cur_value_by_article_model = cur_value_by_overall_model
				if self.a_fit[article_id]:
					cur_value_by_article_model = self.a_models[article_id].predict([user])

				cur_value = cur_value_by_overall_model + cur_value_by_article_model

				if best_value < cur_value:
					best_value = cur_value
					selected_article = article_id
					same_values = list([article_id])

				elif best_value == cur_value:
					same_values.append(article_id)

			selected_article = np.random.choice(same_values, 1)
			
		return selected_article, warmup