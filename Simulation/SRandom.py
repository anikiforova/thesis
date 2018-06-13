import numpy as np
import math
from numpy.linalg import inv
from random import randint

from SBase import Regressor

class SRandom:
	
	def __init__(self, alpha, users, regressor = Regressor.NoRegressor):
		np.random.seed(seed=3432)
		self.alpha = alpha
		self.clusters = len(users)

	def get_alpha(self):
		return self.alpha

	def update(self, user_id, click):
		pass

	def select(self):
		return  randint(0, self.clusters-1)





