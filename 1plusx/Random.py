import numpy as np
import math
from numpy.linalg import inv
from random import randint

from AlgoBase import AlgoBase

class Random(AlgoBase):
	
	def __init__(self, meta):
		super(Random, self).__init__(meta)

	def update(self, users, clicks):
		# self.prediction = np.random.uniform(0, 1, self.user_count)
		self.prediction = np.random.randint(0, 2, self.user_count)
		


