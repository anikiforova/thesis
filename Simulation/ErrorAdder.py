from enum import Enum
import numpy as np
import random 

class ErrorType (Enum):
	Standard_Error = 0,
	Swap_5_Percent = 0,

def add_error(click_estimators):
	length = len(click_estimators)
	error = np.random.normal(0, 0.001, length)
	click_estimators = click_estimators + error

	return click_estimators	


def evaluate_clicks(click_estimators, limit_value):
	return click_estimators >= limit_value

def randomize_clicks(clicks, to_randomize_clicks):
	if to_randomize_clicks:
		percent_clicks_to_randomize = 0.1

		total_click_count = np.count_nonzero(clicks)
		
		index_count_to_change = int(total_click_count * percent_clicks_to_randomize)

		clicks_indexes = np.where(clicks == 1)[0]
		clicks_indexes_to_reverse = np.random.choice(clicks_indexes, index_count_to_change)

		noclicks_indexes = np.where(clicks == 0)[0]
		noclicks_indexes_to_reverse = np.random.choice(noclicks_indexes, index_count_to_change)

		clicks[clicks_indexes_to_reverse] = 0
		clicks[noclicks_indexes_to_reverse] = 1

	return clicks

