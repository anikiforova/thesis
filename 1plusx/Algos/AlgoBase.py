import numpy as np
import math
import random
from random import randint
from sklearn import linear_model
from enum import Enum

class AlgoBase:
	
	def __init__(self, meta):
		self.meta = meta
		self.testMeta = ""
		if self.meta.initialize_user_embeddings:
			self.update_user_embeddings()

	# to ensure inheritance of the method
	def update_single_prediction(self, embedding, campaign_id):
		pass

	def update_multi_campaign_predictions(self):
		print("Updating predictions..")
		for index, embedding in enumerate(self.user_embeddings):
			
			best_normalized_estimate = 0

			for campaign_id in self.campaign_ids:
				ctr_estimate, normalized_estimate = self.update_single_prediction(embedding, campaign_id)

				if normalized_estimate > best_normalized_estimate:
					best_normalized_estimate = normalized_estimate

					self.prediction[index] = ctr_estimate
					self.campaign_assignment[index] = campaign_id

	def update_user_embeddings(self, embeddings_file_name_postfix = ""):
		print("Updating user embeddings..")
		user_ids, user_embeddings = self.meta.read_user_embeddings(embeddings_file_name_postfix)
		
		self.user_count 	 = len(user_ids)

		arrangement 	= np.arange(self.user_count)
		user_ids 		= user_ids[arrangement]
		user_embeddings = user_embeddings[arrangement]

		self.user_ids = user_ids
		self.user_hash_to_id = dict(zip(user_ids, range(0, len(user_ids))))
		self.user_id_to_hash = dict(zip(range(0, len(user_ids)), user_ids))
		
		self.user_embeddings = user_embeddings

	def reset_predictions(self):
		print("Resetting predictions..")
		self.prediction = np.ones(self.user_count) * 0.002 + np.random.uniform(0, 0.001, self.user_count)		
		self.campaign_assignment = np.zeros(self.user_count)		
		
	def setup(self, testMetadata):
		self.testMeta = testMetadata
		self.clickers = set()
		if self.meta.initialize_user_embeddings:
			self.reset_predictions()
			if self.meta.soft_click:
				self.user_impressions = np.zeros(self.user_count)
		
	def prepareClicks(self, users, clicks):
		new_users = np.array([ self.user_hash_to_id[x] for x in users ])
		scaled_clicks = np.array(clicks)
		if self.meta.soft_click:
			for i in np.arange(0, len(new_users)):
				self.user_impressions[new_users[i]] += 1
				scaled_clicks[i] = 1.0 / float(self.user_impressions[new_users[i]])

		users = new_users
		clicks = scaled_clicks		

		if self.testMeta.click_percent > 0:
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

			click_sample_size 	= int(total * self.testMeta.click_percent)
			total_click_size 	= click_sample_size + click_count
			total 				= no_click_count + total_click_size

			print("Total click: {0} No Click {1} Click Sample Size {2} TotalClickers {3}(includes warmup)".format(click_count, no_click_count, click_sample_size, len(self.clickers)))
		
			click_sample 	= np.random.choice(np.arange(0, len(self.clickers)), 	click_sample_size, 		True)
			#no_click_sample = np.random.choice(np.arange(0, no_click_count), 		no_click_sample_size, 	False)

			batch_user_click_sample = np.array(list(self.clickers))[click_sample]

			if click_count > 0:
				batch_users_click = np.append(new_users_click, batch_user_click_sample)
			else:
				batch_users_click = batch_user_click_sample

			users 	= np.append(batch_users_click, new_users_no_click)		
			clicks 	= np.append(np.ones(total_click_size), np.zeros(no_click_count)).reshape([total, 1])
		
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

	def getAssignment(self, user_hash):
		user_id = self.user_hash_to_id[user_hash]
		return self.campaign_assignment[user_id]

	def get_recommendations(self, percent):
		count = int(self.user_count * percent)
		recommendation_ids = self.prediction.argsort()[-count:][::-1]
		# for i in np.arange(0, 5):
		# 	print("R: {} ID: {}".format(self.prediction[recommendation_ids[i]], recommendation_ids[i]))

#		print("Best prediction:" + str(self.prediction[recommendation_ids[0]]))
		recommendation_hashes = set([ self.user_id_to_hash[x] for x in recommendation_ids ])

		return recommendation_hashes








