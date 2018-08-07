import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq

campaign_id = "722100"
path = "../../RawData/Campaigns/{0}/"
processed_path = "../../RawData/Campaigns/{0}/Processed/{1}"

full_path = path.format(campaign_id)
file_path = full_path + "/{0}"

all_files = glob.glob(full_path + "user*.parquet")
all_files.sort()


users_file_name = "all_users.csv"
output = open(processed_path.format(campaign_id, users_file_name), "w")
output.write("UserEmbedding,UserHash\n")

for file_name in all_files:
	file_index = file_name.split("/")[-1].split(".")[0]

	print("Reading file {0}".format(file_name))
	file = pd.read_parquet(file_name, engine='pyarrow')
	data = file.rename(index=str, columns={"v":"UserEmbedding", "hash":"UserHash"})

	users = data.groupby(['UserHash'])['UserEmbedding', 'UserHash'].head(1)
		
	np.set_printoptions(precision=4)
	users['UserEmbedding'] = users['UserEmbedding'].apply(lambda x: np.array(x))
	users = users.values

	for u in users:
		user_str = np.array2string(u[0], separator=' ')[1:-1].replace('\n', '')
		output.write("{0},{1}\n".format(user_str, u[1]))
		output.flush()

output.close()

	
