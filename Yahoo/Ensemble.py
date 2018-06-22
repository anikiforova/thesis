import numpy as np
import math
import random

from EGreedy_Disjoint import EGreedy_Disjoint
from TS_Disjoint import TS_Disjoint
from LinUCB_Disjoint import LinUCB_Disjoint
# from UCB_Seg import UCB_Seg
from AlgorithmType import AlgorithmType

class Ensemble:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.number_of_segments = 5

		self.algos = list()
		# self.algos.append(EGreedy_Disjoint(0.1))
		# self.algos.append(EGreedy_Disjoint(0.15))
		# self.algos.append(TS_Disjoint(0.1))
		# self.algos.append(TS_Disjoint(0.15))
		self.algos.append(LinUCB_Disjoint(0.1))
		self.algos.append(LinUCB_Disjoint(0.15))
		self.algos.append(LinUCB_Disjoint(0.2))

		# self.algos.append(Combo_Seg(0.1, ))
		# self.algos.append(UCB_Seg(0.05))

	def update(self, user, selected_article, click):
		for algo in self.algos:
			algo.update(user, selected_article, click)
		
	def warmup(self, fo):
		pass
		
	def select(self, user, pre_selected_article, lines, total_impressions, click):
		model_id = random.randint(0, len(self.algos)-1)	

		return self.algos[model_id].select(user, pre_selected_article, lines, total_impressions, click)

