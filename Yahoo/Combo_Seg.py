import numpy as np
import math
import random

from EGreedy import EGreedy
from EGreedy_Lin import EGreedy_Lin
from UCB import UCB

from AlgorithmType import AlgorithmType

class Combo_Seg:
	
	def __init__(self, alpha, algorithm_type):
		self.alpha = alpha
		self.number_of_segments = 5

		self.segments_count = dict()
		self.segments_count[0] = 0
		self.segments_count[1] = 0
		self.segments_count[2] = 0
		self.segments_count[3] = 0
		self.segments_count[4] = 0

		self.segments = list()
		for i in range(0, self.number_of_segments): 
			if algorithm_type == AlgorithmType.UCB:
				scale_training_size = 1.0/self.number_of_segments
				self.segments.append(UCB(self.alpha, scale_training_size))
			elif algorithm_type == AlgorithmType.EGreedy:
				self.segments.append(EGreedy(self.alpha))
			elif algorithm_type == AlgorithmType.EGreedy_Lin:
				self.segments.append(EGreedy_Lin(self.alpha))
			else :
				raise NotImplementedError("Non-implemented algorithm." + algorithm_type.name)

	def get_user_segment(self, user):
		user_segment_id = -1
		max_value = 0
		for i in range(0, self.number_of_segments):
			if max_value < user[i]:
				max_value = user[i]
				user_segment_id = i
		return user_segment_id

	def update(self, user, selected_article, click):
		segment_id = self.get_user_segment(user)
		self.segments[segment_id].update(user, selected_article, click)
		
	def warmup(self, fo):
		pass
		
	def select(self, user, pre_selected_article, lines, total_impressions, click):
		segment_id = self.get_user_segment(user)
		self.segments_count[segment_id] +=  1
		selected_article, warmup = self.segments[segment_id].select(user, pre_selected_article, lines, total_impressions, click)
		return selected_article, warmup

