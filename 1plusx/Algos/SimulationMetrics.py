import math
import numpy as np

def get_simulation_metrics(s_clicks, s_predicted_ctr, background_ctr):
	impression_count 	= len(s_clicks)

	cur_ctr = np.mean(s_clicks)
	Calibration = cur_ctr / background_ctr
	
	ne_base = 0.0
	i = 0
	for c, v in zip(s_clicks, s_predicted_ctr):
		if v <= 0 or v >= 1: continue
		ne_base += c*math.log(v) + (1-c)*math.log(1-v)
		if i < 5:
			i += 1
			print("C:{},V:{:.04},ne_base:{:.04}".format(c, v, c*math.log(v) + (1-c)*math.log(1-v)))

	background_ctr_ne = background_ctr * np.log(background_ctr) + (1 - background_ctr) * np.log(1 - background_ctr) 
	NE = (ne_base / impression_count) / background_ctr_ne

	return cur_ctr, Calibration, NE, 1.0 - NE 


def get_model_metrics(background_impressions, model_impressions):
	clicks 		= background_impressions == 1 
	no_clicks 	= background_impressions == 0

	P = np.sum(clicks)
	N = np.sum(no_clicks)

	s_clicks 	= model_impressions == 1
	s_no_clicks = model_impressions == 0

	TP = np.sum(clicks & s_clicks)
	FN = np.sum(clicks & s_no_clicks)
	FP = np.sum(no_clicks & s_clicks)  
	TN = np.sum(no_clicks & s_no_clicks)  

	TPR = TP / P # true positive rate
	FNR = FN / P # true positive rate
	FPR = FP / N # false positive rate
	PPR = TP / (TP + FP) # positive prediction rate

	ROC = TPR / FPR

	MSE = ((background_impressions-model_impressions)**2).mean(axis=None)

	return {"MSE": MSE, 
			"ROC": ROC, 
			"TPR": TPR, 
			"FNR": FNR, 
			"FPR": FPR, 
			"PPR": PPR, 
			"TP" : TP,
			"FN" : FN,
			"FP" : FP,
			"TN" : TN}










