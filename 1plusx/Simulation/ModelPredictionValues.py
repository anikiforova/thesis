import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv

import sys
from os import path
# to be able to access sister folders
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import Algos.Util as Util
from Algos.Metadata import Metadata 
from Algos.Regression import Regression 
from Algos.TestMetadata import TestMetadata 


folder_name = 0
path = "../../RawData/Campaigns/"
impressions_file_path_extension = "/Processed/sorted_time_impressions.csv"
user_file_path_extension = "/Processed/all_users.csv"
predictions_file_path_extension = "/Processed/predictions.csv"

campaign_id = 722100
print ("Starting model for {}..".format(campaign_id))
impressions_file_path = "{0}/{1}/{2}".format(path, campaign_id, impressions_file_path_extension)
user_file_path 		  = "{0}/{1}/{2}".format(path, campaign_id, user_file_path_extension)

impressions = read_csv(impressions_file_path, ",")

users = impressions["UserHash"].values
clicks = impressions["Click"].values

meta = Metadata(campaign_id)
algo = Regression(meta)
testMeta = TestMetadata(meta)
testMeta.click_percent = 0.0
algo.setup(testMeta)
algo.fit(users, clicks)

print("Reading all user embeddings...")
meta = Metadata(folder_name)
user_ids, user_embeddings = meta.read_user_embeddings()

print("Starting prediction for {0}".format(campaign_id))
prediction = algo.model.predict(user_embeddings)

output_path = "{0}/{1}/{2}".format(path, campaign_id, predictions_file_path_extension)
np.savetxt(output_path, prediction, delimiter="\n")







