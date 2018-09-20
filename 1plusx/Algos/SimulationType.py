from enum import Enum

class SimulationType (Enum):
	HINDSIGHT 	= 0
	LOWER		= 1

def get_friendly_name(type):
	if type == SimulationType.HINDSIGHT:
		return "Hindsight"
	else:
		return "Lower"

