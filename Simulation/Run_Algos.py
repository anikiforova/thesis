import time
import datetime
import random
import numpy as np
import pandas as pd
import statistics as stats
import re
import math

from numpy.linalg import inv

# from Util import to_vector
from AlgoFactory import AlgoFactory
from AlgorithmType import AlgorithmType

random.seed(9999)
total_lines = 4681992.0

alphas = {
		AlgorithmType.Random: 			np.arange(0.05, 0.1, 0.05), # no point in different alphas
		AlgorithmType.EFirst:			np.arange(0.05, 0.3, 0.05),
		AlgorithmType.EGreedy:			np.arange(0.05, 0.3, 0.05),
		AlgorithmType.LinUCB_Disjoint:	np.arange(0.05, 0.3, 0.05), # starts decreasing around at 0.25 
		AlgorithmType.LinUCB_GP:		np.arange(0.05, 0.1, 0.05),
		AlgorithmType.LinUCB_GP_All:	np.arange(0.05, 0.1, 0.05),
		AlgorithmType.LinUCB_Hybrid:	np.arange(0.05, 0.3, 0.05),
		AlgorithmType.UCB:				np.arange(0.05, 0.2, 0.05), # limit to only 1 since same value for different alphas
		AlgorithmType.EGreedy_Seg:		np.arange(0.05, 0.3, 0.05),
		AlgorithmType.EGreedy_Disjoint:	np.arange(0.05, 0.3, 0.05), 
		AlgorithmType.EGreedy_Hybrid:	np.arange(0.05, 0.3, 0.05), 
		AlgorithmType.UCB_Seg:			np.arange(0.05, 0.1, 0.05) # limit to only 1 since same value for different alphas

}

choice = AlgorithmType.UCB

output = open('./Results/{0}.csv'.format(choice.name), "w")
output.write("Clicks, Impressions, Alpha, Method\n")	

for alpha in alphas[choice]:
	print('Starting evaluation of {0} with {1}'.format(choice,alpha))

	algo = AlgoFactory.get_algorithm(choice, alpha, 100)

	fo = open("./Data.csv", "r")
	fo.readline()
	
	total_impressions = 0.0
	click_count = 0.0
	impression_count = 0.0

	for line in fo:
		total_impressions += 1
		line = line.split(",")
		user = np.fromstring(line[0], sep=" ")
		user = user / np.linalg.norm(user)
		pre_selected = np.argmax(user)
		click = int(line[2])

		# print(len(user))
		selected, explore = algo.select(user, pre_selected, click)
		# print( selected)
		# print(selected_article)
		if selected == pre_selected:
			print('.', end='', flush=True)
			click_count += click
			algo.update(user, selected, click)

			impression_count += 1
		
			if impression_count % 1000 == 0:
				print('{:.2%} Explore {:.3%}'.format(total_impressions/total_lines, click_count/impression_count))
				output.write('{:d},{:d},{:.2f},{}\n'.format(int(click_count), int(impression_count), alpha, choice.name))
				output.flush()

	output.write('{:d},{:d},{:.2f},{}\n'.format(int(click_count), int(impression_count), alpha, choice.name))
	output.flush()
	fo.close()	

output.close()