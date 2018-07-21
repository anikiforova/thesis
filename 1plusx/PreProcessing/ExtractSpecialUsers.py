import numpy as np
import random 
import math
import time
import datetime

from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt

alphas 						= [0.02] #[0.001, 0.005, 0.01, 0.02, 0.05]
user_recommendation_part 	= [0.05] 
user_train_part 			= [0.05]

print_output 				= True
print_mse 					= False

path = "..//..//RawData//Campaigns//809153//Processed"
input = open("{0}//sorted_time_impressions.csv".format(path), "r")
output = open("{0}//special_users.csv".format(path), "w")
input.readline() # get rid of header
line = input.readline()
hour_begin_timestamp = datetime.datetime.fromtimestamp(int(line.split(",")[2])/1000)

# algo.setup()

special = set()
for total_impressions, line in enumerate(input):
	parts = line.split(",")
	user_id = int(parts[0])
	click = int(parts[1])
	timestamp_raw = int(parts[2])/1000
	timestamp = datetime.datetime.fromtimestamp(timestamp_raw)
	
	
	if timestamp_raw >= 1529568283 and timestamp_raw <= 1529653613:
		special.add(user_id)


input.close()
		
output.write("UserHash\n")
for user in special:
	output.write("{0}\n".format(user))
output.close()


















