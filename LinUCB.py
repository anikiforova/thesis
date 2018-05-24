import random
import numpy as np
import pandas as pd
import statistics as stats
import re
from numpy.linalg import inv
import math

def to_vector(input):
	input = re.split("[: ]", input)

	return np.array([float(input[2]), float(input[4]), float(input[6]), float(input[8]), float(input[10]), float(input[12])])

choice = "LibUCB_Lin"
fo = open("..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")
output = open("..//R6//20090501_" + choice + "_ctr_timeline.csv", "w")
output.write("Impressions, Clicks\n")

# dimensions
d = 6
alpha = 0.4
total_lines = 4681992

random.seed(9999)

# list of Aa 
context = dict()
inverted_context = dict()
beta = dict()

joined_impressions = 0
joined_clicks = 0
total_impressions = 0.0
exploit = False
for line in fo:
	total_impressions = total_impressions  + 1
	# Train on 10% of the data
	bucket = random.randint(0,10)
	exploit = bucket != 0

	line = line.split("|")
	no_space_line = line[0].split(" ")
	pre_selected_article = int(no_space_line[1])
	click = int(no_space_line[2])
	user = to_vector(line[1])

	limit = 0
	selected_article = -1

	for i in range(2, len(line)):
		article_id = int(line[i].split(" ")[0])
		
		if article_id not in context:
			#article = to_vector(line[i])
			context[article_id] = np.identity(d)
			inverted_context[article_id] = np.identity(d)
			beta[article_id] = np.zeros(d)

		cur_inverted_context = inverted_context.get(article_id)
		cur_beta = beta.get(article_id)
		cur_theta = cur_inverted_context.dot(cur_beta)
		if exploit:
			cur_limit = cur_theta.T.dot(user) 
		else:
			cur_limit = cur_theta.T.dot(user) + 0.1 * math.sqrt(user.reshape([1, d]).dot(cur_inverted_context).dot(user)) 

		if(cur_limit > limit):
			selected_article = article_id
			limit = cur_limit

	if selected_article != pre_selected_article:
		continue

	if exploit:
		percent = '{:.1%}'.format(total_impressions/total_lines)
		print(percent + " " + str(selected_article) + " " + str(limit))
		joined_impressions = joined_impressions + 1
		joined_clicks = joined_clicks + click 

	updated_context = context.get(selected_article) + user.reshape([d,1]).dot(user.reshape([1,d]))
	context[selected_article] = updated_context
	inverted_context[selected_article] = inv(updated_context)
	beta[selected_article] = beta.get(selected_article) + click*user

	# break;
	if joined_impressions % 100 == 0 and exploit:
		output.write(str(joined_impressions) + ", " + str(joined_clicks) + "\n")
		output.flush()

output.close()
fo.close()	