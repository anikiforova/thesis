import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv

campaign_ids = [722100, 597165, 837817]#, 809153]
folder_name = 0

path = "../../RawData/Campaigns/"
user_file_path_extension = "/Processed/all_users.csv"

all_users = pd.DataFrame(columns=list(["UserEmbedding", "UserHash"]))

for campaign_id in campaign_ids:
	user_file_path = "{0}/{1}/{2}".format(path, campaign_id, user_file_path_extension)
	cur_users = read_csv(user_file_path, ",")
	cur_count = cur_users.shape[0]
	all_users = all_users.append(cur_users, ignore_index=True)
	
	before = all_users.shape[0]
	print(before)	
	all_users = all_users.groupby(['UserHash'])['UserEmbedding', 'UserHash'].head(1)
	after = all_users.shape[0]
	print(after)
	print("Overlap:" + str(before - after) + " of " + str(cur_count) + " - " + str((before - after) / cur_count))	

output = open("{0}/{1}/{2}".format(path, folder_name, user_file_path_extension), "w")
output.write("UserEmbedding,UserHash\n")
for index, row in all_users.iterrows():
	output.write("{0},{1}\n".format(row["UserEmbedding"], row["UserHash"]))
	output.flush()

output.close()