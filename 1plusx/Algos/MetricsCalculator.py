import math
import numpy as np

def get_entropy_metrics(s_clicks, s_predicted_values, background_ctr, print=False):
	impression_count 	= len(s_clicks)

	cur_ctr = np.mean(s_clicks)
	Calibration = cur_ctr / background_ctr
	
	ne_base = 0.0
	i = 0
	for c, v in zip(s_clicks, s_predicted_values):
		if v <= 0 or v >= 1: continue
		ne_base += c*math.log(v) + (1-c)*math.log(1-v)
		if i < 5 and print:
			i += 1
			print("C:{},V:{:.04},ne_base:{:.04}".format(c, v, c*math.log(v) + (1-c)*math.log(1-v)))

	background_ctr_ne = background_ctr * np.log(background_ctr) + (1 - background_ctr) * np.log(1 - background_ctr) 
	NE = (ne_base / impression_count) / background_ctr_ne

	return {"CTR"			: cur_ctr,
			"Calibration"	: Calibration,
			"NE"			: NE,
			"RIG"			: 1 - NE}

def get_iterative_model_metrics(background_impressions, model_background_predictions_values, model_impressions, model_batch_impressions):
	mi = np.array(model_impressions)
	mbi = np.array(model_batch_impressions)
	model_ctr 		= np.mean(mi)
	model_batch_ctr = np.mean(mbi)

	clicks 			= np.sum(mi == 1)
	impressions 	= len(mi)

	bi = np.array(background_impressions)
	mbpv = np.array(model_background_predictions_values)
	MSE = np.mean((bi-mbpv)**2)

	background_clicks_filter 	= bi == 1
	background_no_clicks_filter = bi == 0

	MMSE_1 = np.mean((bi[background_clicks_filter]-mbpv[background_clicks_filter])**2)
	MMSE_0 = np.mean(((bi[background_no_clicks_filter]-mbpv[background_no_clicks_filter]) * math.sqrt(300))**2)
	MMSE = MMSE_0 + MMSE_1

	return {"ModelCTR"		: model_ctr,
			"ModelBatchCTR"	: model_batch_ctr,
			"MSE"			: MSE,
			"MMSE"			: MMSE,
			"Clicks"		: clicks,
			"Impressions"	: impressions}

def get_full_model_metrics(background_impressions, model_impressions):
	bi = np.array(background_impressions)
	mi = np.array(model_impressions)

	clicks 		= bi == 1 
	no_clicks 	= bi == 0

	P = np.sum(clicks)
	N = np.sum(no_clicks)

	s_clicks 	= mi == 1
	s_no_clicks = mi == 0

	TP = np.sum(clicks & s_clicks)
	FN = np.sum(clicks & s_no_clicks)
	FP = np.sum(no_clicks & s_clicks)  
	TN = np.sum(no_clicks & s_no_clicks)  

	TPR = TP / P # true positive rate
	FNR = FN / P # true positive rate
	FPR = FP / N # false positive rate
	
	PPR = np.nan
	ROC = np.nan

	if TP+FP != 0:
		PPR = TP / (TP + FP) # positive prediction rate

	if FPR != 0:
		ROC = TPR / FPR

	MSE = np.mean((bi-mi)**2)
	# here multiplication by 300 is enough since it's 0/1 values
	MMSE = np.mean((bi-mi * math.sqrt(300))**2)

	return {"MSE": MSE,
			"MMSE":MMSE, 
			"ROC": ROC, 
			"TPR": TPR, 
			"FNR": FNR, 
			"FPR": FPR, 
			"PPR": PPR, 
			"TP" : TP,
			"FN" : FN,
			"FP" : FP,
			"TN" : TN}










