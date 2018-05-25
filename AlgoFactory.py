from enum import Enum

from LinUCB_Hybrid import LinUCB_Hybrid 
from LinUCB_Disjoint import LinUCB_Disjoint
from Random import Random
from EFirst import EFirst
from EGreedy import EGreedy
from util import to_vector

class AlgorithmType (Enum):
	Random = 0
	EFirst = 1
	EGreedy = 2
	LinUCB_Disjoint = 3
	LinUCB_Hybrid = 4

class AlgoFactory:

	def get_algorithm(algorithm_type, alpha):
		if algorithm_type == AlgorithmType.Random:
			return Random(alpha)
		elif algorithm_type == AlgorithmType.EFirst:
			return EFirst(alpha)
		elif algorithm_type == AlgorithmType.EGreedy:
			return EGreedy(alpha) 
		elif algorithm_type == AlgorithmType.LinUCB_Disjoint:
			return LinUCB_Disjoint(alpha)
		elif algorithm_type == AlgorithmType.LinUCB_Hybrid:
			return LinUCB_Hybrid(alpha)
		else:
			print("Non-implemented algorithm type")
