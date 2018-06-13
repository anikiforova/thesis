
import numpy as np
import random 
import pandas
from pandas import read_csv
from pandas import DataFrame


infile_pattern = "../../1plusx/part-000{:02}-9492a20a-812b-4f35-92fa-8f8d9aca22e4-c000.csv"  
outfile_pattern = "../../1plusx/spart-000{:02}-9492a20a-812b-4f35-92fa-8f8d9aca22e4-c000.csv"  


for file_index in range(2, 12):
	output = open(outfile_pattern.format(file_index), "w")

	users = read_csv(infile_pattern.format(file_index), sep=',', header='infer')
	users = users.values

	count = 0
	for user in users:
		user = np.fromstring(user[0][1:-1], sep=",")
		user_str = np.array2string(user, precision=2, separator=',', suppress_small=True)[1:-1].replace('\n', '')
		output.write(user_str + "\n")
		if count % 1000 == 0:
			print('.', end='', flush=True)
			output.flush()
		count += 1

	# input.close()
	output.close()
	print("Done with: {0}".format(file_index))

