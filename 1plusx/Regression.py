import numpy as np
import math
import random
from random import randint
from sklearn import linear_model
from enum import Enum

from AlgoBase import AlgoBase

class Regression(AlgoBase):
	
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, filter_clickers = False, soft_click = False):
		super(Regression, self).__init__(user_embeddings, user_ids, filter_clickers, soft_click)

	def setup(self, alpha):
		super(Regression, self).setup(alpha)
		self.o_users = np.array([], dtype=np.uint32)
		self.o_clicks = np.array([], dtype=np.uint32)

		self.model = linear_model.LinearRegression()

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(Regression, self).prepareClicks(users, clicks)

		self.o_users = np.append(self.o_users, users)
		self.o_clicks = np.append(self.o_clicks, clicks)
		cur_users = self.user_embeddings[self.o_users]
		
		self.model = linear_model.LinearRegression()
		self.model.fit(cur_users, self.o_clicks)
		self.prediction = self.model.predict(self.user_embeddings)

		super(Regression, self).predictionPosprocessing(users, clicks)	
		print(" Done.")

	def get_recommendations(self, count):
		explore = int( self.alpha * count )
		exploit = count - explore

		ordered_predictions = self.prediction.argsort()
		recommendation_ids_exploit = ordered_predictions[-exploit:][::-1]
		recommendation_ids_explore = np.random.choice(ordered_predictions[:exploit], explore, replace= False)

		all_recommendation_ids = np.append(recommendation_ids_exploit, recommendation_ids_explore)
		recommendation_hashes = [ self.user_id_to_hash[x] for x in all_recommendation_ids]

		return set(recommendation_hashes)

