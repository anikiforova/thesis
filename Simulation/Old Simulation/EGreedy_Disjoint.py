import numpy as np
import math
import random
from numpy.linalg import inv
from random import randint

class EGreedy_Disjoint:
	
	def __init__(self, alpha, cluster_count):
		self.d = 100
		self.alpha = alpha
		self.cluster_count = cluster_count
		
		self.A = dict()
		self.A_i = dict()
		self.b = dict()

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

	def select(self, user, pre_selected, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha
		
		selected_article = -1
		
		if explore:
			selected_article = randint(0, self.cluster_count-1)
		else:
			limit = 0.0
			for c in range(0, self.cluster_count):
				cur_limit = user.dot(self.A_i[c].dot(self.b[c]))
				if cur_limit > limit:
					selected_article = c
					limit = cur_limit

		return selected_article, False





