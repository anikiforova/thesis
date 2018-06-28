import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq

campaign_id = "809153"
path = "../../RawData/Campaigns/{0}/"
processed_path = "../../RawData/Campaigns/{0}/Processed/{1}"

full_path = path.format(campaign_id)
file_path = full_path + "/{0}"

all_files = glob.glob(full_path + "*.parquet")
all_files.sort()

add_header = True
for file_name in all_files:
	file_index = file_name.split("/")[-1].split(".")[0]

	print("Reading file {0}".format(file_name))
	file = pd.read_parquet(file_name, engine='pyarrow')
	data = file.rename(index=str, columns={"v":"UserEmbedding", "hash":"UserHash", "kind":"Click"})
	data['Timestamp'] = data['event'].apply(lambda x: x.get('timestamp'))
	data["Click"] = data['Click'].apply(lambda x: ((x == "Click") * 1) + (x == "Conversion") * 2)
	
	data.to_csv(processed_path.format(campaign_id, "time_impressions.csv"), mode="a", sep=",", header=add_header, \
		index=False, columns=["UserHash", "Click", "Timestamp"])

	if int(file_index) <= 9: 
		print("Skip extracting users for file {0}".format(file_index))
		continue
	users = data.groupby(['UserHash'])['UserEmbedding', 'UserHash'].head(1)
		
	np.set_printoptions(precision=4)
	users['UserEmbedding'] = users['UserEmbedding'].apply(lambda x: np.array(x))
	users = users.values

	users_file_name = file_index + "_users.csv"
	output = open(processed_path.format(campaign_id, users_file_name), "w")

	output.write("UserEmbedding,UserHash\n")

	for u in users:
		user_str = np.array2string(u[0], separator=' ')[1:-1].replace('\n', '')
		output.write("{0},{1}\n".format(user_str, u[1]))
		output.flush()

	output.close()
	add_header = False

# # import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# # name = "00030"
# path = "..//..//RawData//Campaigns"
# full_path = "{0}//{1}.parquet".format(path, name)
# parquet_file = pq.ParquetFile(full_path)
	
