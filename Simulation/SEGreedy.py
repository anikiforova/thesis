import numpy as np
import math
import random
from random import randint
from sklearn import linear_model

from SBase import SBase 

class SEGreedy(SBase):
	
	def __init__(self, alpha, users, regressor, total_impressions = 0):
		SBase.__init__(self, alpha, users, regressor)

	def get_explore(self):
		bucket = random.uniform(0, 1)
		return bucket <= self.alpha
		
	def update(self, user_id, click):		
		SBase.update(self, user_id, click)
		

