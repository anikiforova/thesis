import numpy as np
from pandas import read_csv
from pandas import DataFrame

class Metadata:
	soft_click 					= False
	filter_clickers 			= False
	remove_duplicate_no_clicks 	= False

	dimensions 					= 100
	
	cluster_name_postfix		= ""

	total_lines = dict({597165:2257957, 837817:5409623, 722100:8475542, 809153:14392075})
	
	def __init__(self, campaign_id):
		self.campaign_id = campaign_id
		self.path = "..//..//RawData//Campaigns//{0}//Processed".format(self.campaign_id)

	def normalize_array(self, array):
		min_value = np.min(array)
		max_value = np.max(array)
		return (array - min_value)/(max_value - min_value)

	def read_user_embeddings(self):
		print("Reading users.. ", end='', flush=True)	
		users = read_csv("{0}//all_users{1}.csv".format(self.path, self.cluster_name_postfix), header=0)
		print(" Done.")

		print("Parsing users.. ", end='', flush=True)	
		user_ids = np.array(users["UserHash"])
		user_count = len(user_ids)
		user_embeddings = np.array(users["UserEmbedding"])
		user_embeddings = np.array([np.fromstring(x, sep=" ") for x in user_embeddings]).reshape([user_count, self.dimensions])
		print(" Done.")

		print("Normalizing users.. ", end='', flush=True)	
		user_embeddings = np.apply_along_axis(lambda e: self.normalize_array(e), 0, user_embeddings)
		print(" Done.")
		return user_ids, user_embeddings










