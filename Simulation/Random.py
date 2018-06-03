import numpy as np
import math
from numpy.linalg import inv
from random import randint

class Random:
	
	def __init__(self, alpha, clusters):
		np.random.seed(seed=9999)
		self.clusters = clusters

	def update(self, user, selected_article, click):	
		pass

	def select(self, user, pre_selected, click):
		return  randint(0, self.clusters), False





