import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime
from pandas import read_csv

a = pd.date_range(start='26/8/2018', end='26/08/2018')

printHeader = True
for date in a:
	date = date.strftime("%Y-%m-%d")

	print(date)
	path = "../../RawData/Multi-Campaign/Impressions/{0}/"
	
	data = read_csv( "../../RawData/Multi-Campaign/Processed/5_Impressions_{0}.csv".format(date), sep=",")

	clickers = np.unique(np.array(data.loc[result['Click'] == 1]["UserHash"]))

	# group = data.groupby(["UserHash"]).agg({"CampaignId": pd.Series.nunique, "Click": np.size})
	group = data.groupby(["UserHash"])["CampaignId"].agg( [("CampaignCount", pd.Series.nunique), 
				("Impressions", np.size), 
				("CampaignIds", lambda x: "{%s}" % ', '.join(str(a) for a in np.unique(x)))])
	
	group.reset_index(level=0, inplace=True)
	result = group.groupby(["CampaignIds", "CampaignCount"])["UserHash"].agg([("DistinctUserCount", np.size)])
	result.reset_index(level=0, inplace=True)
	result.reset_index(level=0, inplace=True)
	
	userGroup = result.groupby(["CampaignCount"])["DistinctUserCount"].agg([("UserCount", np.sum)])
	total_users = userGroup["UserCount"].sum()
	userGroup["PercentFromTotalUserCount"] = userGroup["UserCount"] / total_users
	userGroup.reset_index(level=0, inplace=True)
	userGroup["Date"] = date
	# print(userGroup)
	# break
	
	campaignUniqueUserCounts = data.groupby(["CampaignId"])["UserHash"].agg( [("TotalDistinctUserCount", pd.Series.nunique)])
	campaignUniqueUserCounts.reset_index(level=0, inplace=True)
	# print(campaignUniqueUserCounts)

	singleCampaignCounts = result.loc[result['CampaignCount'] == 1]
	singleCampaignCounts["CampaignId"] = singleCampaignCounts["CampaignIds"].apply(lambda c: int(c[1:-1]))
	# print(singleCampaignCounts)

	joined = singleCampaignCounts.set_index('CampaignId').join(campaignUniqueUserCounts.set_index('CampaignId'), lsuffix='_caller', rsuffix='_other')
	joined = joined.rename(index=str, columns={"DistinctUserCount": "NonJoinedUserCount"})
	joined.reset_index(level=0, inplace=True)
	joined["Date"] = date

	# print(joined)
	mode = "a"
	if printHeader:
		mode = "w"

	joined.to_csv("../../RawData/Multi-Campaign/Processed/MultiCampaignClickStats.csv", mode=mode, sep=",", \
		header=printHeader, index=False, columns=["CampaignId", "NonJoinedUserCount", "TotalDistinctUserCount", "Date"])	
	
	userGroup.to_csv("../../RawData/Multi-Campaign/Processed/MultiCampaignUserClickStats.csv", mode=mode, sep=",", \
		header=printHeader, index=False, columns=["CampaignCount", "UserCount", "PercentFromTotalUserCount", "Date"])	
	
	printHeader = False
	# break