import numpy as np
import math
from numpy.linalg import inv
from random import randint

from TargetBase import TargetBase
from AlgoBase import AlgoBase

class Random_Multi(AlgoBase, TargetBase):
	
	def __init__(self, meta, campaign_ids, start_date, end_date):
		AlgoBase.__init__(self, meta)
		TargetBase.__init__(self, meta, campaign_ids, start_date, end_date)

	def setup(self, testMeta):
		AlgoBase.setup(self, testMeta)
		TargetBase.setup(self, testMeta)
			
	def update(self, campaign_ids, users, clicks):
		self.prediction = np.random.uniform(0, 1, self.user_count)
		campaign_ids_list = list(self.campaign_ids)
		self.campaign_assignment = [campaign_ids_list[a] for a in np.random.randint(len(self.campaign_ids), size=self.user_count)]
		
	def update_prediction(self):		
		self.prediction = np.random.uniform(0, 1, self.user_count)
		campaign_ids_list = list(self.campaign_ids)
		self.campaign_assignment = [campaign_ids_list[a] for a in np.random.randint(len(self.campaign_ids), size=self.user_count)]
