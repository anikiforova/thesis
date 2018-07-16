
import glob

for name in glob.glob("./Results/TS_Lin_Equal*.csv"):
	nameParts = name.split("/")
	print(name)
	input_file = name
	output_file = "./Results/CTR/" + nameParts[2] 
	
	input = open(input_file, "r")
	line = input.readline() # skip header
	data = line.split(",")
	if len(data) < 6: continue
	output = open(output_file, "w")	

	output.write("Clicks,Impressions,CumClicks,CumImpressions,TotalImpressions,RecommendationSizePercent,Timestamp,Alpha,Method\n")
	print(input_file)

	alpha = ""

	for line in input:
		data = line.split(",")
		# print(line)
		CumClicks 					= float(data[0])
		CumImpressions 				= float(data[1])
		TotalImpressions 			= data[2]
		Method 					  	= data[3]
		RecommendationSizePercent 	= data[4]
		RecommendationSize 			= data[5]
		if len(data) <= 6:
			Timestamp = "NA"
		else:
			Timestamp					= data[6]

		if len(data) <= 7:
			cur_alpha = "NA"
		else:
			cur_alpha 					= data[7][0:-1]
		
		cur_clicks = CumClicks
		cur_impressions = CumImpressions

		if cur_alpha == alpha:
			cur_clicks = CumClicks - prev_clicks
			cur_impressions = CumImpressions - prev_impressions
			prev_clicks = CumClicks
			prev_impressions = CumImpressions
		else: # beginning of new experiment.
			alpha = cur_alpha
			prev_clicks = CumClicks
			prev_impressions = CumImpressions

		output.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(cur_clicks, cur_impressions, CumClicks, CumImpressions, TotalImpressions, RecommendationSizePercent, Timestamp, cur_alpha, Method))

	input.close()
	output.close()
