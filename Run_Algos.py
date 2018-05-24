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
output = open("..//R6//20090501_" + choice + "_ctr_timeline.csv", "w")
output.write("Explore_Clicks, Explore_Impressions, Exploit_Clicks, Exploit_Impressions\n")

# dimensions
algo = LinUCB_Hybrid()
total_lines = 4681992.0

random.seed(9999)

total_impressions = 0.0
exploit = False
explore_count = 0.0
exploit_count = 0.0
explore_click = 0.0
exploit_click = 0.0

for line in fo:

	total_impressions += 1
	# Train on 10% of the data
	bucket = random.randint(0,10)
	exploit = bucket != 0

	line = line.split("|")
	no_space_line = line[0].split(" ")
	pre_selected_article = int(no_space_line[1])
	click = int(no_space_line[2])
	user = to_vector(line[1])


	selected_article = algo.select(user, line[2:], exploit)

	if selected_article == pre_selected_article:
		algo.update(user, selected_article, click)

		if exploit:
			exploit_click += click
			exploit_count += 1
		else:
			explore_click += click
			explore_count += 1
		
		if (explore_count + exploit_count) % 100 == 0 and exploit:
			percent = '{:.2%}'.format(total_impressions/total_lines)
			explore_state = '{:.3%}'.format(explore_click/explore_count)
			exploit_state = '{:.3%}'.format(exploit_click/exploit_count)
			print(percent + " Explore: " + explore_state + " Exploit " + str(exploit_state))

			output.write(str(explore_click) + ", " + str(explore_count) + ", " + str(exploit_click) + "," + str(exploit_count) + "\n")
			output.flush()

output.close()
fo.close()	