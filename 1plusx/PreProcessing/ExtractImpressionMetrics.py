import pandas as pd
import os 
import glob
import numpy as np
from pandas import read_csv
from pandas import DataFrame
import pyarrow.parquet as pq

impressions =  read_csv("../../RawData/Campaigns/5/Processed/all_impressions.csv", ",")

impressions["Timestamp"] = (impressions["Timestamp"]/(3600 * 1000 * 4)).astype(int)
impressions["Timestamp"] *= (3600 * 1000 * 4) 
# print(impressions.dtypes)

group = impressions.groupby(["CampaignId", "Timestamp"])

result = group.agg({"Click": {"Impressions": np.size, "Clicks":np.sum, "CTR": np.mean}})

columnNames = dict({ "Click_Impressions": 	"Impressions", 
					 "Click_Clicks": 		"Clicks", 
					 "Click_CTR": 			"CTR"})

result.columns = [columnNames["_".join(x)] for x in result.columns.ravel()]
# print(result.columns)
result.to_csv("../../RawData/Campaigns/5/Processed/all_impressions_agg.csv")
