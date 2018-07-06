
import glob

for name in glob.glob("../Results/*.csv"):
	nameParts = name.split("/")
	input_file = name
	output_file = "../Results/CTR/" + nameParts[3] 
	
	input = open(input_file, "r")
	output = open(output_file, "w")

	line = input.readline() # skip header
	
	output.write("Clicks,Impressions,CumClicks,CumImpressions,TotalImpressions,RecommendationSizePercent,Timestamp,Alpha,Method\n")
	print(input_file)

	alpha = ""

	for line in input:
		data = line.split(",")
		CumClicks 					= int(data[0])
		CumImpressions 				= int(data[1])
		TotalImpressions 			= data[2]
		Method 					  	= data[3]
		RecommendationSizePercent 	= data[4]
		RecommendationSize 			= data[5]
		Timestamp					= data[6]
		cur_alpha 					= data[7]
		
		cur_clicks = CumClicks
		cur_impressions = CumImpressions

		if cur_alpha == alpha:
			cur_clicks = CumClicks - prev_clicks
			cur_impressions = CumImpressions - prev_impressions
			prev_clicks = cum_clicks
			prev_impressions = cum_impressions
		else: # beginning of new experiment.
			alpha = cur_alpha
			prev_clicks = CumClicks
			prev_impressions = CumImpressions

		output.write("{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(cur_clicks, cur_impressions, CumClicks, CumImpressions, TotalImpressions, RecommendationSizePercent, Timestamp, Alpha, Method))

	input.close()
	output.close()
