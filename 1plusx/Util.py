import time
import datetime

def get_line_info(line):
	parts 			= line.split(",")
	user_id 		= int(parts[0])
	click 			= int(parts[1])
	timestamp_raw 	= int(parts[2])/1000
	timestamp 		= datetime.datetime.fromtimestamp(timestamp_raw)

	return user_id, click, timestamp_raw, timestamp