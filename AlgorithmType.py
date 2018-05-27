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