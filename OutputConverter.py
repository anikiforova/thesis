
import glob


for name in glob.glob("./Yahoo/Results/*.csv"):
	nameParts = name.split("/")
	input_file = name
	output_file = "./Yahoo/Results/CTR/" + nameParts[3] 
	
	input = open(input_file, "r")
	output = open(output_file, "w")

	line = input.readline() # skip header
	output.write("Clicks, Impressions, CumImpressions, Alpha, Method\n")
	print(input_file)

	alpha = ""

	for line in input:
		data = line.split(",")
		if data[2] == alpha:
			cum_clicks = int(data[0])
			cum_impressions = int(data[1])
			cur_clicks = cum_clicks - prev_clicks
			cur_impressions = cum_impressions - prev_impressions
			output.write("{0},{1},{2},{3},{4}".format(cur_clicks, cur_impressions, cum_impressions, data[2], data[3]))
			prev_clicks = cum_clicks
			prev_impressions = cum_impressions
		else:
			alpha = data[2]
			prev_clicks = int(data[0])
			prev_impressions = int(data[1])
			output.write("{0},{1},{2},{3},{4}".format(prev_clicks, prev_impressions, prev_impressions, data[2], data[3]))

	input.close()
	output.close()
