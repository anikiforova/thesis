import numpy as np
import math
import random
from random import randint
from sklearn import linear_model

class SEGreedy:
	
	def __init__(self, alpha, users):
		self.alpha = alpha
		self.users = users
		self.user_count = len(self.users)

		self.bucket_size = 1000

		# observed
		self.o_users = np.array([], dtype=np.uint32)
		self.o_clicks = np.array([], dtype=np.uint32)

		self.predition = np.ones(self.user_count) * 0.05

		self.mask = np.array(np.ones(self.user_count), dtype=bool)

	def update(self, user_id, click):
		# print(user_id)
		self.o_users = np.append(self.o_users, user_id)
		self.o_clicks = np.append(self.o_clicks, click)

		if len(self.o_users) % self.bucket_size == 0:
			# print(self.o_users)
			cur_users = self.users[self.o_users]
			model = linear_model.LinearRegression()
			model.fit(cur_users, self.o_clicks)
			self.predition = model.predict(self.users)

	def select(self):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha

		found = True
		while found:
			if explore:
				selected_user = randint(0, self.user_count-1)
			else:
				max_value_excluding_previously_chosen = np.amax(self.predition[self.mask])
				# print("Value: " + str(self.mask[0]))
				selected_users = np.argwhere(self.predition == max_value_excluding_previously_chosen)

				if len(selected_users) > 1:
					selected_user = selected_users[randint(0, len(selected_users)-1)][0]
				else:
					selected_user = selected_users[0]
			
			selected_user = int(selected_user)

			if self.mask[selected_user] == 1:
				self.mask[selected_user] = 0	
				found = False
			# print(selected_user)
		return selected_user

