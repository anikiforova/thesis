import numpy as np
import math
import random
from random import randint
from sklearn import linear_model
from enum import Enum

from .AlgoBase import AlgoBase
# from AlgoBase import AlgoBase

class Regression(AlgoBase):
	
	def __init__(self, meta):
		super(Regression, self).__init__(meta)

	def setup(self, testMeta):
		super(Regression, self).setup(testMeta)
		self.o_users = np.array([], dtype=np.uint32)
		self.o_clicks = np.array([], dtype=np.uint32)

		self.model = linear_model.LinearRegression()

	def fit(self, users, clicks):
		print("Starting Fit.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks)
		self.o_users = np.append(self.o_users, users)
		self.o_clicks = np.append(self.o_clicks, clicks)
		cur_users = self.user_embeddings[self.o_users]
		
		self.model = linear_model.LinearRegression()
		self.model.fit(cur_users, self.o_clicks)
		
	def predict_now(self, user_embedding):
		cur_prediction = self.model.predict([user_embedding])
		return cur_prediction[0]

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		self.fit(users, clicks)
		self.prediction = self.model.predict(self.user_embeddings)
		self.predictionPosprocessing(users, clicks)	
		print(" Done.")

	def get_recommendations(self, percent):
		count = int(self.user_count * percent)
		explore = int( self.testMeta.alpha * count )
		exploit = count - explore

		ordered_predictions = self.prediction.argsort()
		recommendation_ids_exploit = ordered_predictions[-exploit:][::-1]
		recommendation_ids_explore = np.random.choice(ordered_predictions[:exploit], explore, replace= False)

		all_recommendation_ids = np.append(recommendation_ids_exploit, recommendation_ids_explore)
		recommendation_hashes = [ self.user_id_to_hash[x] for x in all_recommendation_ids]

		return set(recommendation_hashes)

