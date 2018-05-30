import numpy as np

input = open('./Data.csv', "r")
input.readline() # skip header

for line in input:
	line_split = line.split(",")
	user = np.fromstring(line_split[0], dtype=float, sep=' ')
	ad = np.fromstring(line_split[1], dtype=float, sep=' ')
	click = int(line_split[2])
	
	
print(clicks/impressions)
input.close()