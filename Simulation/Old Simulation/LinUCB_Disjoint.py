import numpy as np
import math
from numpy.linalg import inv

class LinUCB_Disjoint:
	
	def __init__(self, alpha, cluster_count):
		self.d = 100
		self.alpha = alpha
		self.training_size = 1000
		self.warmup_impressions = 0
		self.cluster_count = cluster_count

		self.A = dict()
		self.A_i = dict()
		self.b = dict()
		self.L = dict()

		for c in range(0, self.cluster_count):
			self.A[c] = np.identity(self.d)
			self.A_i[c] = np.identity(self.d)
			self.b[c] = np.zeros(self.d)
			

	def update(self, user, selected_article, click):
		self.A[selected_article] += user.reshape([self.d, 1]).dot(user.reshape([1, self.d]))
		self.A_i[selected_article] = inv(self.A[selected_article])
		if click == 1 :
			self.b[selected_article] += user
	
	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, click):
		selected_article = -1
		warmup = False

		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			self.update(user, pre_selected_article, click)
			selected_article = pre_selected_article
			warmup = True
		else:
			limit = 0.0
			for c in range(0, self.cluster_count):
				cur_A_i = self.A_i[c]
				cur_b = self.b[c]

				cur_theta = cur_A_i.dot(cur_b)
				cur_limit = user.dot(cur_theta) + self.alpha * math.sqrt(user.reshape([1, self.d]).dot(cur_A_i).dot(user))

				if(cur_limit > limit):
					selected_article = c
					limit = cur_limit
		return selected_article, warmup





