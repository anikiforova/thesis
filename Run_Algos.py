import datetime
import random
import numpy as np
import pandas as pd
import statistics as stats
import re
from numpy.linalg import inv
import math

from util import to_vector
from AlgoFactory import AlgoFactory
from AlgoFactory import AlgorithmType

random.seed(9999)
total_lines = 4681992.0
alphas = np.arange(0.05, 0.3, 0.05)
# factory = AlgoFactory()

choice = AlgorithmType.LinUCB_Hybrid

today = '%d_%m'.format(datetime.date.today())
output = open('./Results/20090501_{0}_{1}.csv'.format(choice, today), "w")
output.write("Clicks, Impressions, Alpha, Method\n")	

for alpha in alphas:
	print('Starting evaluation of {0} with {1}'.format(choice,alpha))

	algo = AlgoFactory.get_algorithm(choice, alpha)

	fo = open("..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")
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

		selected_article = algo.select(user, line[2:], total_impressions)

		if selected_article == pre_selected_article:
			# print('.', end='', flush=True)
			algo.update(user, pre_selected_article, click)
			click_count += click
			impression_count += 1
		
			if impression_count % 1000 == 0:
				print('{:.2%} Explore {:.3%}'.format(total_impressions/total_lines, click_count/impression_count))
				output.write('{:d},{:d},{:.2f},{}\n'.format(int(click_count), int(impression_count), alpha, choice))
				output.flush()
	fo.close()	


output.close()