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

class SBase:
	
	def __init__(self, alpha, users, regressor = Regressor.LinearRegression, bucket_size = 100):
		self.alpha = alpha
		self.users = users
		self.user_count = len(self.users)

		self.bucket_size = 100

		self.new_users = np.array([], dtype=np.uint32)
		self.new_clicks = np.array([], dtype=np.uint32)

		self.o_users = np.array([], dtype=np.uint32)
		self.o_clicks = np.array([], dtype=np.uint32)

		self.predition = np.ones(self.user_count) * 0.05

		self.mask = np.array(np.ones(self.user_count), dtype=bool)
	
		self.regressor = regressor
		if self.regressor == Regressor.LinearRegression:
			self.model = linear_model.LinearRegression()
		else:	
			self.model = linear_model.SGDClassifier(loss='hinge', penalty='l2')


	def get_alpha(self):
		return self.alpha
			
	def get_explore(self):
		pass

	def update(self, user_id, click):
		self.new_users = np.append(self.new_users, user_id)
		self.new_clicks = np.append(self.new_clicks, click)
		# self.o_users = np.append(self.o_users, user_id)
		# self.o_clicks = np.append(self.o_clicks, click)

		if len(self.new_users) % self.bucket_size == 0:
			# print(self.regressor)
			if self.regressor == Regressor.LinearRegression:
				self.o_users = np.append(self.o_users, self.new_users)
				self.o_clicks = np.append(self.o_clicks, self.new_clicks)
				# print(len(self.o_clicks))
				cur_users = self.users[self.o_users]
				
				self.model = linear_model.LinearRegression()
				self.model.fit(cur_users, self.o_clicks)

			else:
				cur_users = self.users[self.new_users]
				# cur_users = self.users[self.o_users]

				self.model = self.model.partial_fit(cur_users, self.new_clicks, [0, 1])
				
			self.predition = self.model.predict(self.users)

			self.new_users = np.array([], dtype=np.uint32)
			self.new_clicks = np.array([], dtype=np.uint32)	

	def select(self):
		explore = self.get_explore()

		found = True
		while found:
			if explore:
				selected_user = randint(0, self.user_count-1)
			else:
				max_value_excluding_previously_chosen = np.amax(self.predition[self.mask])
				selected_users = np.argwhere(self.predition == max_value_excluding_previously_chosen)

				if len(selected_users) > 1:
					selected_user = selected_users[randint(0, len(selected_users)-1)][0]
				else:
					selected_user = selected_users[0]
			
			selected_user = int(selected_user)

			if self.mask[selected_user] == 1:
				self.mask[selected_user] = 0	
				found = False

		return selected_user

