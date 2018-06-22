import numpy as np
import math
import random

from EGreedy import EGreedy 

class EGreedy_Annealing(EGreedy):

	def __init__(self, alpha):
		self.initial_alpha = float(alpha)
		EGreedy.__init__(self, alpha)

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		EGreedy.alpha = float(self.initial_alpha) / float(self.initial_alpha + total_impressions)

		return EGreedy.select(self, user, pre_selected_article, lines, total_impressions, click)
