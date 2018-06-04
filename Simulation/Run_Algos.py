import time
import datetime
import random
from random import randint
import numpy as np
import pandas as pd
import re
import math
import sys

from numpy.linalg import inv
# from termcolor import colored

# from Util import to_vector
from AlgoFactory import AlgoFactory
from AlgorithmType import AlgorithmType
from AlgorithmType import get_algorithm_type

random.seed(9999)
total_lines = 635387.0 # per file - 12 files

alphas = {
		AlgorithmType.Random: 			np.arange(0.05, 0.1, 0.05), # no point in different alphas
		# AlgorithmType.EFirst:			np.arange(0.05, 0.1, 0.05),
		AlgorithmType.EGreedy:			np.arange(0.001, 0.3, 0.05),
		# AlgorithmType.LinUCB_Disjoint:	np.arange(0.05, 0.1, 0.05), # starts decreasing around at 0.25 
		# AlgorithmType.LinUCB_GP:		np.arange(0.05, 0.1, 0.05),
		# AlgorithmType.LinUCB_GP_All:	np.arange(0.05, 0.1, 0.05),
		# AlgorithmType.LinUCB_Hybrid:	np.arange(0.05, 0.1, 0.05),
		AlgorithmType.UCB:				np.arange(0.001, 0.3, 0.05), # limit to only 1 since same value for different alphas
		# AlgorithmType.EGreedy_Seg:		np.arange(0.05, 0.1, 0.05),
		AlgorithmType.EGreedy_Disjoint:	np.arange(0.001, 0.3, 0.05), 
		# AlgorithmType.EGreedy_Hybrid:	np.arange(0.05, 0.1, 0.05), 
		# AlgorithmType.UCB_Seg:			np.arange(0.05, 0.1, 0.05) # limit to only 1 since same value for different alphas

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
		print('Starting evaluation of {0} with {1}'.format(choice, alpha))

		algo = AlgoFactory.get_algorithm(choice, alpha, 100)

		total_impressions = 0.0
		click_count = 0.0
		impression_count = 0.0

		for file_index in range(0, 12):
			fo = open("../../1plusx/clicks_part-000{:02}-9492a20a-812b-4f35-92fa-8f8d9aca22e4-c000.csv".format(file_index), "r")
			fo.readline()
			
			for line in fo:
				total_impressions += 1
				line = line.split(",")
				user = np.fromstring(line[0], sep=" ")
				user = user / np.linalg.norm(user)
				pre_selected = int(randint(0, len(user)-1))
				# print(user)
				# print(pre_selected)
				click = int(line[2])

				# print(len(user))
				selected, explore = algo.select(user, pre_selected, click)
				# print(str(pre_selected) + " " + str(selected))
				
				if selected == pre_selected:
					# print(str(pre_selected) + " " + str(selected))
					# print('.', end='', flush=True)
					click_count += click
					algo.update(user, selected, click)

					impression_count += 1
				
					if impression_count % 100 == 0:
						print('{:.2%} Explore {:.3%}'.format(total_impressions/total_lines, click_count/impression_count))
						output.write('{:d},{:d},{:.2f},{}\n'.format(int(click_count), int(impression_count), alpha, choice.name))
						output.flush()

			print('{:.2%} Explore {:.3%}'.format(total_impressions/total_lines, click_count/impression_count))
			output.write('{:d},{:d},{:.2f},{}\n'.format(int(click_count), int(impression_count), alpha, choice.name))
			output.flush()
			fo.close()		

	output.close()