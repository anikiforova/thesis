import numpy as np
import math
from numpy.linalg import inv
from random import randint

# from SBase import Regressor

class Random:
	
	def __init__(self, alpha, user_embeddings, user_ids, dimensions, filter_clickers = False, soft_click = False):
		np.random.seed(seed=3432)
		self.alpha = alpha
		self.user_ids = user_ids
		self.indexes = np.arange(0, len(self.user_ids)-1)

	def get_alpha(self):
		return self.alpha

	def warmup(self, input, warmup_impression_count):
		pass

	def update(self, users, clicks):
		pass

	def get_recommendations(self, count):
		choice = np.random.choice(self.indexes, count, replace=False)
		return set(self.user_ids[choice])





