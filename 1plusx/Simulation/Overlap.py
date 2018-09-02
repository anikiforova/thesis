import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from pandas import read_csv


campaign_ids = [722100, 597165, 837817]#, 809153]
folder_name = 0

path = "../../RawData/Campaigns/"
impressions_file_path_extension = "/Processed/sorted_time_impressions.csv"
user_file_path_extension = "/Processed/all_users.csv"

all_impressions = read_csv("{0}/{1}/{2}".format(path, folder_name, impressions_file_path_extension), ",")
user_hash = np.unique(all_impressions["UserHash"].values)

all_users = read_csv("{0}/{1}/{2}".format(path, folder_name, user_file_path_extension), ",")
user_hash_users = all_users["UserHash"].values


intersection = np.intersect1d(user_hash, user_hash_users)
print("Total Impr UserHash:{}, User UserHash:{}, Intersection:{}".format(len(user_hash), len(user_hash_users), len(intersection)))