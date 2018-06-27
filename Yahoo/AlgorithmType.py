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
	#LinUCB_SGP = 12
	EGreedy_Lin = 13
	EGreedy_Seg_Lin = 14
	EGreedy_Lin_Hybrid = 15
	TS = 16
	TS_Lin = 17
	TS_Seg = 18
	TS_Disjoint = 19
	TS_Hybrid = 20
	TS_Bootstrap = 21
	TS_Truncated = 22
	EGreedy_TS = 23
	TS_Gibbs = 24
	TS_Laplace = 25
	EGreedy_Annealing = 26
	NN = 27
	Ensemble = 28
	TS_RLR = 29

def get_algorithm_type(algorithm_type_string):
	for name, member in AlgorithmType.__members__.items():
		if algorithm_type_string.lower() == name.lower():
			return member
	return -1

