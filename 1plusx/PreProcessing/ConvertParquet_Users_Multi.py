import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime

a = pd.date_range(start='22/8/2018', end='28/08/2018')
for date in a:
	date = date.strftime("%Y-%m-%d")

	path = "../../RawData/Campaigns/10/Users/{0}/"
	
	full_path = path.format(date)
	file_path = full_path + "/{0}"

	all_files = glob.glob(full_path + "*.parquet")
	all_files.sort()

	output = open("../../RawData/Campaigns/10/Processed/Users_{0}.csv".format(date), "w")
	output.write("UserEmbedding,UserHash\n")

	for file_name in all_files:
		file_index = file_name.split("/")[-1].split(".")[0]

		print("Reading file {0}".format(file_name))
		data = pq.read_table(file_name).to_pandas()
		np.set_printoptions(precision=4)
		
		for index, row in data.iterrows():
			user_str = np.array2string(np.array(row['v']), separator=' ')[1:-1].replace('\n', '')
			output.write("{0},{1}\n".format(user_str, row['hash']))
			if index % 10000 == 0:
				output.flush()

	output.close()

