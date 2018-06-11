from enum import Enum

class AlgorithmType (Enum):
	Random = 0
	EFirst = 1
	EGreedy = 2
	LinUCB_Disjoint = 3
	LinUCB_Hybrid = 4
	UCB = 5
	EGreedy_Seg = 6
	UCB_Seg = 7
	EGreedy_Disjoint = 8
	EGreedy_Hybrid = 9
	LinUCB_GP = 10
	LinUCB_GP_All = 11


def get_algorithm_type(algorithm_type_string):
	for name, member in AlgorithmType.__members__.items():
		if algorithm_type_string.lower() == name.lower():
			return member
	return -1


