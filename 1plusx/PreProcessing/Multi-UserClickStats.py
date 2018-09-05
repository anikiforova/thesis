import pandas as pd
import os 
import glob
import numpy as np
import pyarrow.parquet as pq
from datetime import datetime
from pandas import read_csv

a = pd.date_range(start='15/8/2018', end='28/08/2018')

printHeader = True
for date in a:
	date = date.strftime("%Y-%m-%d")

	print(date)
	path = "../../RawData/Multi-Campaign/Impressions/{0}/"
	
	data = read_csv( "../../RawData/Multi-Campaign/Processed/5_Impressions_{0}.csv".format(date), sep=",")
	click_data = data.loc[data['Click'] == 1]
	clickers = np.unique(np.array(click_data["UserHash"]))
	# print(click_data)

	# group = data.groupby(["UserHash"]).agg({"CampaignId": pd.Series.nunique, "Click": np.size})
	group = click_data.groupby(["UserHash"])["CampaignId"].agg( [("CampaignCount", pd.Series.nunique), 
				("CampaignIds", lambda x: "{%s}" % ', '.join(str(a) for a in np.unique(x)))])
	
	group.reset_index(level=0, inplace=True)
	result = group.groupby(["CampaignIds", "CampaignCount"])["UserHash"].agg([("DistinctUserCount", np.size)])
	result.reset_index(level=0, inplace=True)
	result.reset_index(level=0, inplace=True)
	#print(result)

	userGroup = result.groupby(["CampaignCount"])["DistinctUserCount"].agg([("ClickUserCount", np.sum)])
	userGroup.reset_index(level=0, inplace=True)
	total_users = userGroup["ClickUserCount"].sum()
	userGroup["PercentFromTotalClickUserCount"] = userGroup["ClickUserCount"] / total_users
	userGroup["Date"] = date
	print(userGroup)
		
	# # 2. User, # campaigns that it observed, # campaigns they clicked
	clickers_data = data[data["UserHash"].isin(clickers)]

	clickers_group = clickers_data.groupby(["UserHash", "CampaignId"]).agg( {"Click":np.sum})
	clickers_group.reset_index(level=0, inplace=True)
	clickers_group.reset_index(level=0, inplace=True)
	clickers_group["HasClick"] = clickers_group["Click"].apply(lambda a: int(a > 0))
	
	user_campaign_clicks = clickers_group.groupby(["UserHash"]).agg( {"CampaignId": pd.Series.nunique, "HasClick": np.sum})
	user_campaign_clicks.reset_index(level=0, inplace=True)
	user_campaign_clicks.reset_index(level=0, inplace=True)
	user_stats = user_campaign_clicks.groupby(["CampaignId", "HasClick"]).agg({"UserHash":np.size})
	user_stats.reset_index(level=0, inplace=True)
	user_stats.reset_index(level=0, inplace=True)
	user_stats = user_stats.rename(index=str, columns={"HasClick": "ClickedCampaignsCount", "UserHash":"UserCount", "CampaignId": "CampaignCount"})
	user_stats["Date"] = date
	print(user_stats)
	mode = "a"
	if printHeader:
		mode = "w"

	user_stats.to_csv("../../RawData/Multi-Campaign/Processed/MultiCampaignClickStats.csv", mode=mode, sep=",", \
		header=printHeader, index=False, columns=["CampaignCount", "ClickedCampaignsCount", "UserCount", "Date"])	
	
	# Users that clicked on more than 1 campaign
	userGroup.to_csv("../../RawData/Multi-Campaign/Processed/MultiCampaignUserClickStats.csv", mode=mode, sep=",", \
		header=printHeader, index=False, columns=["CampaignCount", "ClickUserCount", "PercentFromTotalClickUserCount", "Date"])	
	
	printHeader = False
	# break