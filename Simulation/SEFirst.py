import numpy as np
import math
import random
from random import randint
from sklearn import linear_model

from SBase import SBase 

class SEFirst(SBase):
	
	def __init__(self, alpha, users, regressor, total_impressions):
		SBase.__init__(self, alpha, users, regressor)

		self.impression_count = 0
		self.total_impressions = total_impressions

	def get_explore(self):
		return self.impression_count <= self.total_impressions * self.alpha		

	def update(self, user_id, click):		
		self.impression_count += 1
		SBase.update(self, user_id, click)
		
