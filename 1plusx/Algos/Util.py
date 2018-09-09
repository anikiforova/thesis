import time
import datetime

def get_line_info(line):
	parts 			= line.split(",")
	user_id 		= int(parts[0])
	click 			= int(parts[1])
	timestamp_raw 	= int(parts[2])
	timestamp 		= datetime.datetime.fromtimestamp(timestamp_raw/1000)

	return user_id, click, timestamp_raw, timestamp

def get_campaign_line_info(line):
	parts 			= line.split(",")
	campaign_id 	= int(parts[0])
	user_id 		= int(parts[1])
	click 			= int(parts[2])
	timestamp_raw 	= int(parts[3])
	timestamp 		= datetime.datetime.fromtimestamp(timestamp_raw/1000)

	return campaign_id, user_id, click, timestamp_raw, timestamp


def get_multi_line_info(line, campaign_ids):
	parts 					= line.split(",")
	user_id 				= int(parts[0])
	timestamp_raw 			= int(parts[1])
	timestamp 				= datetime.datetime.fromtimestamp(timestamp_raw/1000)
	campaign_clicks 		= [int(x) for x in parts[2:2+len(campaign_ids)]]
	campaign_clicks_dict 	= dict(zip(campaign_ids, campaign_clicks))
	
	return user_id, campaign_clicks_dict, timestamp_raw, timestamp