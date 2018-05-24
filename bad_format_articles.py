import random
import numpy as np
import pandas as pd
import statistics as stats
import re
from numpy.linalg import inv
import math

from LinUCB_Hybrid import LinUCB_Hybrid 
from LinUCB_Disjoint import LinUCB_Disjoint

def to_vector(input):
	input = re.split("[: ]", input)

	return np.array([float(input[2]), float(input[4]), float(input[6]), float(input[8]), float(input[10]), float(input[12])])

choice = "LinUCB_Hybrid"
fo = open("..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")

total_impressions = 0.0

# dimensions
articles = dict()
for line in fo:
	lines = line.split("|")
	no_space_line = lines[0].split(" ")
	pre_selected_article = int(no_space_line[1])
	click = int(no_space_line[2])
	user = to_vector(lines[1])

	for cur_line in lines[2:]:
		article_id = int(cur_line.split(" ")[0])
		if article_id not in articles:
			try:
				article = to_vector(cur_line)
				articles[article_id] = "a"
			except IndexError:
				print(line)
				break; 

	if total_impressions % 1000 == 0 and exploit:
		percent = '{:.2%}'.format(total_impressions/total_lines)
		explore_state = '{:.3%}'.format(explore_click/explore_count)
		exploit_state = '{:.3%}'.format(exploit_click/exploit_count)
		print(percent + " Explore: " + explore_state + " Exploit " + str(exploit_state))

fo.close()	
