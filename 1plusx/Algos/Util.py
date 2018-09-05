import time
import datetime

def get_line_info(line):
	parts 			= line.split(",")
	user_id 		= int(parts[0])
	click 			= int(parts[1])
	timestamp_raw 	= int(parts[2])
	timestamp 		= datetime.datetime.fromtimestamp(timestamp_raw/1000)

	return user_id, click, timestamp_raw, timestamp

def get_multi_line_info(line):
	parts 			= line.split(",")
	campaign_id 	= int(parts[0])
	user_id 		= int(parts[1])
	click 			= int(parts[2])
	timestamp_raw 	= int(parts[3])
	timestamp 		= datetime.datetime.fromtimestamp(timestamp_raw/1000)

	return campaign_id, user_id, click, timestamp_raw, timestamp