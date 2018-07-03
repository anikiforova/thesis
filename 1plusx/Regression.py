import numpy as np
import math
import random
from random import randint
from sklearn import linear_model
from enum import Enum

class Regressor(Enum):
	LinearRegression = 0
	SGDClassifier = 1
	NoRegressor = 2

class Regression:
	
	def __init__(self, alpha, user_embeddings, user_ids, regressor = Regressor.LinearRegression, filter_clickers = false):
		self.alpha = alpha
		# users['UserEmbedding']
		self.user_hast_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))
		self.user_count = len(self.user_hast_to_id)
		self.user_embeddings = user_embeddings
		
		self.indexes = np.arange(0, self.user_count-1)

		self.o_users = np.array([], dtype=np.uint32)
		self.o_clicks = np.array([], dtype=np.uint32)

		self.predition = np.ones(self.user_count) * 0.02

		self.regressor = regressor
		if self.regressor == Regressor.LinearRegression:
			self.model = linear_model.LinearRegression()
		else:	
			self.model = linear_model.SGDClassifier(loss='hinge', penalty='l2')

		self.clickers = set()

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		new_users = [ self.user_hast_to_id[x] for x in users ]

		new_click_users = new_users[clicks == 1]
		for user in new_click_users: self.clickers.add(user)

		self.o_users = np.append(self.o_users, new_users)
		self.o_clicks = np.append(self.o_clicks, clicks)
		cur_users = self.user_embeddings[self.o_users]
		
		self.model = linear_model.LinearRegression()
		self.model.fit(cur_users, self.o_clicks)
	
		if filter_clickers:
			user_index_to_use = [index for index in self.indexes if index not in self.clickers] 
			embeddings_to_use = self.user_embeddings[user_index_to_use]
			self.predition = self.model.predict(embeddings_to_use)
		else:
			self.predition = self.model.predict(self.user_embeddings)
		print(" Done.")

	def get_recommendations(self, count):
		recommendation_ids = self.predition.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)

