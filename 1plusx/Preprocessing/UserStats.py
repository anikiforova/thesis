import numpy as np
import random 
import math
import time
import datetime
from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt


class User:
	Id 						= -1
	
	FirstTimestamp			= -1
	FirstClickTimestamp		= -1
	LastTimestamp		 	= -1

	TotalImpressions	 	= -1
	FirstClickIndex			= -1
	ClickCount 				= 0 
	TimeUntilClick			= -1

	def __init__(self, id, timestamp):
		self.Id = id
		self.FirstTimestamp = timestamp
		self.LastTimestamp = timestamp
		self.TotalImpressions = 0

name = "809153"
path = "..//..//RawData//Campaigns"
impressions_path = "{0}//{1}//Processed//sorted_time_impressions.csv".format(path, name)
user_stats_path = "{0}//{1}//Processed//user_statistics.csv".format(path, name)

print("Processing impression data..", end='', flush=True)
input = open(impressions_path, "r")
input.readline()

user_stats = dict()
for line in input:
	parts = line.split(",")
	user_id = int(parts[0])
	click = int(parts[1])
	timestamp = int(parts[2])/1000
	# timestamp = datetime.datetime.fromtimestamp(timestamp_raw)

	if user_id not in user_stats.keys():
		user_stats[user_id] = User(user_id, timestamp)

	if click:
		if user_stats[user_id].FirstClickIndex == -1:
			user_stats[user_id].FirstClickIndex = user_stats[user_id].TotalImpressions
			user_stats[user_id].FirstClickTimestamp = timestamp
			user_stats[user_id].TimeUntilClick = timestamp - user_stats[user_id].FirstTimestamp

		user_stats[user_id].ClickCount += 1
		
	user_stats[user_id].LastTimestamp = timestamp
	user_stats[user_id].TotalImpressions += 1

input.close()
print("Done.")

print("Outputing data to file...", end='', flush=True)
output = open(user_stats_path, "w")
# output.write("Id,FirstTimestamp,FirstClickTimestamp,LastTimestamp,TotalImpressions,FirstClickIndex,ClickImpressionsIndex,ClickCount\n")
output.write("TotalImpressions,FirstClickIndex,ClickCount,TimeUntilClickSec,ActiveTime\n")
for user_id, user in user_stats.items():
	# output.write("{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:s},{:.0f}\n".format(user.Id, user.FirstTimestamp, user.FirstClickTimestamp, user.LastTimestamp, user.TotalImpressions, user.FirstClickIndex, click_impressions_str, len(user.ClickImpressionsIndex)))
	output.write("{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n".format(user.TotalImpressions, user.FirstClickIndex, user.ClickCount, user.TimeUntilClick, user.LastTimestamp - user.FirstTimestamp))
	output.flush()

output.close()
print("Done.")