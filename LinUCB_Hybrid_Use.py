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

choice = "LibUCB_Hybrid"
fo = open("..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")
output = open("..//R6//20090501_" + choice + "_ctr_timeline.csv", "w")
output_error = open("..//R6//20090501_exceptions.txt", "w")
output.write("Explore_Clicks, Explore_Impressions, Exploit_Clicks, Exploit_Impressions\n")

# dimensions
d = 6
k = 36
alpha = 0.4
total_lines = 4681992.0

random.seed(9999)

# list of Aa 
A = dict()
A_i = dict()
B = dict()
b = dict()
articles = dict()

total_impressions = 0.0
exploit = False
explore_count = 0.0
exploit_count = 0.0
explore_click = 0.0
exploit_click = 0.0

A0 = np.identity(k)
b0 = np.zeros(k)
A0_i = np.identity(k)
beta = A0_i.dot(b0)

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
		
		if article_id not in A:
			try:
				article = to_vector(line[i])
			except IndexError:
				output_error.write(line[i])
				continue; 
			articles[article_id] = article
			A[article_id] = np.identity(d)
			A_i[article_id] = np.identity(d)
			B[article_id] = np.zeros(d*k).reshape([d,k])
			b[article_id] = np.zeros(d)

		z = np.outer(user, article).reshape([1, k])
		cur_A = A.get(article_id)
		cur_A_i = A_i.get(article_id)
		cur_B = B.get(article_id)
		cur_b = b.get(article_id)
		
		cur_theta = cur_A_i.dot((cur_b - cur_B.dot(beta))) 
		pre_user_A_i = user.dot(cur_A_i) 
		pre_zT_A0_i = z.dot(A0_i)
		pre_A_i_user = cur_A_i.dot(user)
		
		cur_s = pre_zT_A0_i.dot(z.reshape([k,1])) - 2 * pre_zT_A0_i.dot(cur_B.T).dot(pre_A_i_user) + pre_user_A_i.dot(user) + pre_user_A_i.dot(cur_B).dot(A0_i).dot(cur_B.T).dot(pre_A_i_user)

		if exploit:
			cur_limit = z.dot(beta) + user.dot(cur_theta)
		else:
			cur_limit = z.dot(beta) + user.dot(cur_theta) + alpha * math.sqrt(cur_s)

		# print(cur_limit)
		if(cur_limit > limit):
			selected_article = article_id
			limit = cur_limit

	# break;
	if selected_article != pre_selected_article:
		continue

	article = articles[selected_article]
	z = np.outer(user, article)
	z_1_k = z.reshape([1, k])
	z_k_1 = z.reshape([k, 1])
	user_1_d = user.reshape([1, d])
	user_d_1 = user.reshape([d, 1])
	A[selected_article] = A[selected_article] + user_d_1.dot(user_1_d) 
	A_i[selected_article] = inv(A[selected_article])
	B[selected_article] = B[selected_article] + user_d_1.dot(z_1_k)
	b[selected_article] = b[selected_article] + click * user
	A0 = A0 + z_k_1.dot(z_1_k)
	A0_i = inv(A0)
	bo = b0 + click * z_k_1
	beta = A0_i.dot(b0)

	if exploit:
		exploit_click = exploit_click + click
		exploit_count = exploit_count + 1
		
	else:
		explore_click = explore_click + click
		explore_count = explore_count + 1
	
	if (explore_count + exploit_count) % 100 == 0 and exploit:
		percent = '{:.2%}'.format((explore_count + exploit_count)/total_lines)
		explore_state = '{:.3%}'.format(explore_click/explore_count)
		exploit_state = '{:.3%}'.format(exploit_click/exploit_count)
		print(percent + " Explore: " + explore_state + " Exploit " + str(exploit_state))

		output.write(str(explore_click) + ", " + str(explore_count) + ", " + str(exploit_click) + "," + str(exploit_count) + "\n")
		output.flush()

output_error.close()
output.close()
fo.close()	