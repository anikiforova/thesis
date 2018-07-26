import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq

campaign_id = "837817"
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
	data = file.rename(index=str, columns={"hash":"UserHash", "click":"Click", "timestamp":"Timestamp"})

	#data['Timestamp'] = data['event'].apply(lambda x: x.get('timestamp'))
	data["Click"] = data['Click'].apply(lambda x: ((x == "Click") * 1) + (x == "Conversion") * 2)
	
	data.to_csv(processed_path.format(campaign_id, "time_impressions.csv"), mode="a", sep=",", header=add_header, \
		index=False, columns=["UserHash", "Click", "Timestamp"])

	add_header = False

