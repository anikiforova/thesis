import pandas as pd
import os 
import glob
import numpy as np
from pandas import read_csv 
import pyarrow.parquet as pq

campaign_id = "809153"
users_processed_path = "../../RawData/Campaigns/{0}/Processed/"

full_path = users_processed_path.format(campaign_id)

all_files = glob.glob(full_path + "0*_users.csv")
all_files.sort()

add_header = True
all_users = pd.DataFrame(columns=list(["UserEmbedding", "UserHash"]))

for file_name in all_files:
	print("Reading file {0}".format(file_name))
	# parquet_file = pq.ParquetFile(file_name)
	# print(parquet_file.schema)
	
	data = read_csv(file_name, ",")
	all_users = all_users.append(data, ignore_index=True)
	
	print(all_users.shape)	
	all_users = all_users.groupby(['UserHash'])['UserEmbedding', 'UserHash'].head(1)

output = open(users_processed_path.format(campaign_id) + "all_users.csv", "w")
output.write("UserEmbedding,UserHash\n")

for index, row in all_users.iterrows():
	# user_str = np.array2string(u[0], separator=' ')[1:-1].replace('\n', '')
	output.write("{0},{1}\n".format(row["UserEmbedding"], row["UserHash"]))
	output.flush()

output.close()

# # import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# # name = "00030"
# path = "..//..//RawData//Campaigns"
# full_path = "{0}//{1}.parquet".format(path, name)
# parquet_file = pq.ParquetFile(full_path)
	
