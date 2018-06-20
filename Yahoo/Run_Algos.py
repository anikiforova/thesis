import time
import datetime
import random
import numpy as np
import pandas as pd
import statistics as stats
import re
import math
import sys

from numpy.linalg import inv

from Util import to_vector
from AlgoFactory import AlgoFactory
from AlgorithmType import AlgorithmType
from AlgorithmType import get_algorithm_type

random.seed(9999)
total_lines = 4681992.0

alphas = {
		AlgorithmType.Random: 				np.arange(0.05, 0.1, 0.05), # no point in different alphas
		AlgorithmType.EFirst:				np.arange(0.05, 0.3, 0.05),
		AlgorithmType.EGreedy:				np.arange(0.05, 0.3, 0.05),
		AlgorithmType.LinUCB_Disjoint:		np.arange(0.05, 0.3, 0.05), # starts decreasing around at 0.25 
		AlgorithmType.LinUCB_GP:			np.arange(0.05, 0.1, 0.05),
		# AlgorithmType.LinUCB_SGP:			np.arange(0.05, 0.1, 0.05),
		AlgorithmType.LinUCB_GP_All:		np.arange(0.05, 0.1, 0.05),
		AlgorithmType.LinUCB_Hybrid:		np.arange(0.05, 0.3, 0.05),
		AlgorithmType.UCB:					np.arange(0.5,  0.6, 0.05), # limit to only 1 since same value for different alphas	
		AlgorithmType.EGreedy_Seg:			np.arange(0.05, 0.3, 0.05),
		AlgorithmType.EGreedy_Lin:			[0], #np.arange(0.01, 0.05, 0.05),
		AlgorithmType.EGreedy_Lin_Hybrid:	[0], #np.arange(0.01, 0.05, 0.05),
		AlgorithmType.EGreedy_Seg_Lin:		np.arange(0.01, 0.05, 0.05),
		AlgorithmType.EGreedy_Disjoint:		np.arange(0.05, 0.3, 0.05), 
		AlgorithmType.EGreedy_Hybrid:		np.arange(0.05, 0.3, 0.05), 
		AlgorithmType.UCB_Seg:				np.arange(0.05, 0.1, 0.05), # limit to only 1 since same value for different alphas
		AlgorithmType.TS:					[0],#np.arange(0.05, 0.2, 0.05)
		AlgorithmType.TS_Bootstrap:			[0],#np.arange(0.05, 0.2, 0.05)
		AlgorithmType.TS_Lin:				np.arange(-0.005, 0.005, 0.005),#np.arange(0.05, 0.2, 0.05) 
		AlgorithmType.TS_Seg:				[0],#np.arange(0.05, 0.2, 0.05) 
		AlgorithmType.TS_Disjoint:			np.arange(0.95, 1.05, 0.05),#np.arange(0.05, 0.2, 0.05) 
		AlgorithmType.TS_Hybrid:			np.arange(0.95, 1.05, 0.05),
		AlgorithmType.TS_Truncated:			[0],
		AlgorithmType.TS_Gibbs:				[0],
		AlgorithmType.TS_Laplace:			[0],
		AlgorithmType.EGreedy_TS:			np.arange(0.1, 0.5, 0.1),
}

if len(sys.argv) <= 1:
	# print (colored("No AlgorithmType selected. Please select algorithm type.", 'red'))
	print ("No AlgorithmType selected. Please select algorithm type.")
	sys.exit()


for i in range(1, len(sys.argv)):
	
	choice = get_algorithm_type(sys.argv[i])
	if choice == -1:
		# print (colored("Error. No algorithm type:{0}".format(sys.argv[i]), 'red'))
		print ("Error. No algorithm type:{0}".format(sys.argv[i]))
		continue

	output = open('./Results/{0}.csv'.format(choice.name), "w")
	output.write("Clicks, Impressions, Alpha, Method\n")	

	for alpha in alphas[choice]:
		print('Starting evaluation of {0} with {1}'.format(choice,alpha))

		algo = AlgoFactory.get_algorithm(choice, alpha)

		fo = open("..//..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")
		algo.warmup(fo)
		
		total_impressions = 0.0
		click_count = 0.0
		impression_count = 0.0

		for line in fo:
			total_impressions += 1
			line = line.split("|")
			no_space_line = line[0].split(" ")
			pre_selected_article = int(no_space_line[1])
			click = int(no_space_line[2])
			user = to_vector(line[1])

			selected_article, warmup = algo.select(user, pre_selected_article, line[2:], total_impressions, click)
			# print(selected_article)
			if selected_article == pre_selected_article and not warmup:
				# print('.', end='', flush=True)
				algo.update(user, pre_selected_article, click)
		
				click_count += click
				impression_count += 1
			
				if impression_count % 1000 == 0:
					print('{:.2%} Cumulative CTR: {:.3%}'.format(total_impressions/total_lines, click_count/impression_count))
					output.write('{:d},{:d},{:.2f},{}\n'.format(int(click_count), int(impression_count), alpha, choice.name))
					output.flush()
		fo.close()	

	output.close()