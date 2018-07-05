import numpy as np
import math
import random
from random import randint
from sklearn import linear_model
from enum import Enum

class AlgoBase:
	
	def __init__(self, alpha, user_embeddings, user_ids, filter_clickers = False, soft_click = False):
		self.alpha = alpha
		self.filter_clickers = filter_clickers
		self.soft_click = soft_click

		self.user_hast_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))
		self.user_count = len(self.user_hast_to_id)
		self.user_embeddings = user_embeddings
		
		self.predition = np.ones(self.user_count) * 0.02

		self.clickers = set()
		self.user_impressions = np.zeros(self.user_count)

	def prepareClicks(self, users, clicks):
		new_users = [ self.user_hast_to_id[x] for x in users ]
		scaled_clicks = clicks
		if self.soft_click:
			for i in np.arange(0, len(new_users)):
				self.user_impressions[new_users[i]] += 1
				scaled_clicks[i] = 1.0 / float(self.user_impressions[new_users[i]])

		return new_users, scaled_clicks

	def predictionPosprocessing(self, users, clicks):
		if self.filter_clickers:
			users = np.array(users)
			clicks = np.array(clicks)

			new_click_users = users[clicks == 1]
			for user in new_click_users: 
				self.clickers.add(user)

			for user in self.clickers:
				self.predition[user] = 0 # set prediction of clicker to 0 so they don't get selected from now on (unless randomly selected )				

