import numpy as np
import math
from numpy.linalg import inv

class LinUCB_Disjoint:
	
	def __init__(self):
		self.d = 6
		self.alpha = 0.4

		self.A = dict()
		self.A_i = dict()
		self.b = dict()

	def update(self, user, selected_article, click):
		self.A[selected_article] += user.reshape([self.d, 1]).dot(user.reshape([1, self.d]))
		self.A_i[selected_article] = inv(self.A[selected_article])
		if click == 1 :
			self.b[selected_article] += user
		

	def select(self, user, lines, exploit):
		limit = 0.0
		selected_article = -1
		for line in lines:
			article_id = int(line.split(" ")[0])
		
			if article_id not in self.A:
				self.A[article_id] = np.identity(self.d)
				self.A_i[article_id] = np.identity(self.d)
				self.b[article_id] = np.zeros(self.d)

			cur_A_i = self.A_i[article_id]
			cur_b = self.b[article_id]

			cur_theta = cur_A_i.dot(cur_b)

			if exploit:
				cur_limit = user.dot(cur_theta)
			else:
				cur_limit = user.dot(cur_theta) + self.alpha * math.sqrt(user.reshape([1, self.d]).dot(cur_A_i).dot(user))

			if(cur_limit > limit):
				selected_article = article_id
				limit = cur_limit

		return selected_article





