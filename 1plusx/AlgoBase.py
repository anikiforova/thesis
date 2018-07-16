import numpy as np
import math
import random
from random import randint
from sklearn import linear_model
from enum import Enum

class AlgoBase:
	
	def __init__(self, user_embeddings, user_ids, click_percent = 0.5, equalize_clicks = False, filter_clickers = False, soft_click = False):
		self.filter_clickers = filter_clickers
		self.soft_click = soft_click

		self.equalize_clicks = equalize_clicks
		self.click_percent = click_percent

		self.user_hash_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))
		self.user_count = len(self.user_hash_to_id)
		self.user_embeddings = user_embeddings
		
	def setup(self, alpha):
		self.alpha = alpha
		self.clickers = set()
		self.user_impressions = np.zeros(self.user_count)
		self.prediction = np.ones(self.user_count) * 0.02
		
	def prepareClicks(self, users, clicks):
		new_users = np.array([ self.user_hash_to_id[x] for x in users ])
		scaled_clicks = np.array(clicks)
		if self.soft_click:
			for i in np.arange(0, len(new_users)):
				self.user_impressions[new_users[i]] += 1
				scaled_clicks[i] = 1.0 / float(self.user_impressions[new_users[i]])

		users = new_users
		clicks = scaled_clicks		

		if self.equalize_clicks:
			click_indexes 		= clicks == 1
			no_click_indexes 	= clicks == 0
			new_users_click 	= set(users[click_indexes])
			new_users_no_click 	= set(users[no_click_indexes])
			
			for clicker in new_users_click: self.clickers.add(clicker) 
			for clicker in self.clickers: new_users_no_click.discard(clicker) 

			new_users_click 		= np.array(list(new_users_click))
			new_users_no_click 	= np.array(list(new_users_no_click))

			click_count 		= len(new_users_click)
			no_click_count 		= len(new_users_no_click)
			total 				= click_count + no_click_count

			no_click_sample_size= int(total * self.click_percent)
			click_sample_size 	= total - no_click_sample_size - click_count
			total_click_size = click_sample_size + click_count
			total = no_click_sample_size + total_click_size

			print("Total click: {0} No Click {1} Click Sample Size {2} TotalClickers {3}".format(click_count, no_click_count, click_sample_size, len(self.clickers)))
		
			click_sample 	= np.random.choice(np.arange(0, len(self.clickers)), click_sample_size, True)
			no_click_sample = np.random.choice(np.arange(0, no_click_count), no_click_sample_size, False)

			batch_user_click_sample = np.array(list(self.clickers))[click_sample]

			if click_count > 0:
				batch_users_click = np.append(new_users_click, batch_user_click_sample)
			else:
				batch_users_click = batch_user_click_sample

			batch_users_no_click = new_users_no_click[no_click_sample]

			users 	= np.append(batch_users_click, batch_users_no_click)		
			clicks 	= np.append(np.ones(total_click_size), np.zeros(no_click_sample_size)).reshape([total, 1])
		
		return users, clicks

	def predictionPosprocessing(self, users, clicks):
		if self.filter_clickers:
			users = np.array(users)
			clicks = np.array(clicks)

			new_click_users = users[clicks == 1]
			for user in new_click_users: 
				self.clickers.add(user)

			for user in self.clickers:
				self.prediction[user] = 0 # set prediction of clicker to 0 so they don't get selected from now on (unless randomly selected )				

	def getPrediction(self, user_hash):
		user_id = self.user_hash_to_id[user_hash]
		return self.prediction[user_id]

	def get_recommendations(self, count):
		recommendation_ids = self.prediction.argsort()[-count:][::-1]
		print("Best prediction:" + str(self.prediction[recommendation_ids[0]]))
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)