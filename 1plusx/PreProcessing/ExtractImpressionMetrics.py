import pandas as pd
import os 
import glob
import numpy as np
from pandas import read_csv
from pandas import DataFrame
import pyarrow.parquet as pq
import datetime as dt

hours = 1
breakdowns = int(24 / hours)

# print("Starting reading..")
# impressions =  read_csv("../../RawData/Campaigns/5/Processed/sorted_time_impressions.csv", ",")
# print("Starting conversion..")

# impressions["Timestamp"] = (impressions["Timestamp"]/(3600 * 1000 * hours)).astype(int)
# impressions["Timestamp"] *= (3600 * hours * 1000) 
# impressions["Hour"] = impressions["Timestamp"].apply(lambda a: dt.datetime.fromtimestamp(a/1000).hour)
# # print(impressions.dtypes)

# group = impressions.groupby(["CampaignId", "Timestamp", "Hour"])

# result = group.agg({"Click": {"Impressions": np.size, "Clicks":np.sum, "CTR": np.mean}})

# columnNames = dict({ "Click_Impressions": 	"Impressions", 
# 					 "Click_Clicks": 		"Clicks", 
# 					 "Click_CTR": 			"CTR"})

# result.columns = [columnNames["_".join(x)] for x in result.columns.ravel()]

# result.to_csv("../../RawData/Campaigns/5/Processed/{}_impression_breakdown.csv".format(breakdowns))


data = read_csv("../../RawData/Campaigns/5/Processed/{}_impression_breakdown.csv".format(breakdowns), ",")

avg_group = data.groupby(["CampaignId", "Hour"])
avg_result = avg_group.agg({"Impressions":np.mean, "Clicks": np.mean})

avg_result.to_csv("../../RawData/Campaigns/5/Processed/{}_avg_impression_breakdown.csv".format(breakdowns))