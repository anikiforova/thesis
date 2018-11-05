from enum import Enum

class TargetSplitType (Enum):
	NO_SPLIT 	= 0
	DAILY		= 1
	HALF_DAY	= 2

def get_algorithm_type(algorithm_type_string):
	for name, member in TargetSplitType.__members__.items():
		if algorithm_type_string.lower() == name.lower():
			return member
	return -1

