import numpy as np
import math
import re
from numpy.linalg import inv

def to_vector(input):
	input = re.split("[: ]", input)

	return np.array([float(input[2]), float(input[4]), float(input[6]), float(input[8]), float(input[10]), float(input[12])])