import numpy as np
import math
import random
from random import randint
from sklearn import linear_model
from enum import Enum

class AlgoBase:
	
	def __init__(self, meta):
		self.meta = meta
		user_ids, user_embeddings = self.meta.read_user_embeddings()
		
		self.user_count 	 = len(user_ids)

		arrangement = np.arange(self.user_count)
		user_ids 		= user_ids[arrangement]
		user_embeddings = user_embeddings[arrangement]

		self.user_ids = user_ids
		self.user_hash_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))
		
		self.user_embeddings = user_embeddings
		
	def setup(self):
		self.clickers = set()
		self.prediction = np.ones(self.user_count) * 0.02 + np.random.uniform(0, 0.001, self.user_count)

		if self.meta.soft_click:
			self.user_impressions 	= np.zeros(self.user_count)
		
	def prepareClicks(self, users, clicks):
		new_users = np.array([ self.user_hash_to_id[x] for x in users ])
		scaled_clicks = np.array(clicks)
		if self.meta.soft_click:
			for i in np.arange(0, len(new_users)):
				self.user_impressions[new_users[i]] += 1
				scaled_clicks[i] = 1.0 / float(self.user_impressions[new_users[i]])

		users = new_users
		clicks = scaled_clicks		

		if self.meta.equalize_clicks:
			click_indexes 			= clicks == 1
			no_click_indexes 		= clicks == 0
			new_users_click 		= np.array(users[click_indexes])
			new_users_no_click 		= np.array(users[no_click_indexes])
			# new_users_no_click_set 	= set(users[no_click_indexes])
			
			for clicker in new_users_click: self.clickers.add(clicker) 
			# for clicker in self.clickers: new_users_no_click_set.discard(clicker) 

			# new_users_click 		= np.array(list(new_users_click))
			# new_users_no_click 		= np.array(list(new_users_no_click))

			click_count 		= len(new_users_click)
			no_click_count 		= len(new_users_no_click)
			total 				= click_count + no_click_count

			no_click_sample_size= int(total * self.meta.no_click_percent)
			click_sample_size 	= total - no_click_sample_size - click_count
			total_click_size 	= click_sample_size + click_count
			total 				= no_click_sample_size + total_click_size

			print("Total click: {0} No Click {1} Click Sample Size {2} TotalClickers {3}(includes warmup)".format(click_count, no_click_count, click_sample_size, len(self.clickers)))
		
			click_sample 	= np.random.choice(np.arange(0, len(self.clickers)), 	click_sample_size, 		True)
			no_click_sample = np.random.choice(np.arange(0, no_click_count), 		no_click_sample_size, 	False)

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
		if self.meta.filter_clickers:
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

	def get_recommendations(self, percent):
		count = int(self.user_count * percent)
		recommendation_ids = self.prediction.argsort()[-count:][::-1]
#		print("Best prediction:" + str(self.prediction[recommendation_ids[0]]))
		recommendation_hashes = set([ self.user_id_to_hash[x] for x in recommendation_ids ])

		return recommendation_hashes