import numpy as np
import random 
import math
from pandas import read_csv
from pandas import DataFrame

from SEGreedy import SEGreedy 
from SRandom import SRandom 

user_dimension = 100
percentile = 95
update_theta = False #True
update_limit = False

simulation_impressions = 10000.0
error = np.random.normal(0, 0.0001, int(simulation_impressions) + 1 )
poly = 3



input = open("..//..//R6//ydata-fp-td-clicks-v1_0.20090501_single_article.csv", "r")

users = np.array([])
clicks = list([])
for line in input:
	line = line.split(',')
	click = int(line[1])
	user = np.fromstring(line[0], sep=" ")
	users = np.append(users, user)
	clicks.append(click)

users = users.reshape([len(clicks), 6])

input.close()

alpha = 0
algo = SRandom(alpha, users)

# output = open("./Results/Simulation_Results_Poly_Sigmoid.csv", "w")
impression = 0.0
click_count = 0.0
while impression < simulation_impressions:
	selected_user = algo.select()
	# print(selected_user)
	click = clicks[selected_user]
	algo.update(selected_user, click)
	
	click_count += click
	impression += 1

	if impression % 100 == 0:
		# output.write('{:d},{:d},{:.2f},{}\n'.format(int(click_count), int(impression), algo.get_alpha(), algo_name))
		print('Done with: {:.1%} CTR: {:.3%}'.format(impression/simulation_impressions, click_count/impression))
		# output.flush()


# output.close()

