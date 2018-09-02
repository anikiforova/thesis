import numpy as np
from pandas import read_csv
from pandas import DataFrame

class Metadata:
	soft_click 					= False
	filter_clickers 			= False
	remove_duplicate_no_clicks 	= False

	dimensions 					= 100
	
	cluster_name_postfix		= ""
	
	base_path = "../../RawData/Campaigns/"

	total_lines = dict({0:0, 
		597165:2257957, 
		837817:5409623, 
		722100:8475542, 
		809153:14392075,
		847460:3937517,
		856805:12806271,
		858140:4517501,
		865041:4357812,
		866128:4819814})
	
	def __init__(self, campaign_id):
		self.campaign_id = campaign_id
		self.path = "{0}/{1}/Processed".format(self.base_path, self.campaign_id)

	def normalize_array(self, array):
		min_value = np.min(array)
		max_value = np.max(array)
		return (array - min_value)/(max_value - min_value)

	def read_user_embeddings(self):
		print("Reading users.. ", end='', flush=True)	
		filePath = "{0}/all_users{1}.csv".format(self.path, self.cluster_name_postfix)
		print("({})".format(filePath))
		users = read_csv(filePath, header=0)
		
		print("Parsing users.. ", end='', flush=True)	
		user_ids = np.array(users["UserHash"])
		user_count = len(user_ids)
		user_embeddings = np.array(users["UserEmbedding"])
		user_embeddings = np.array([np.fromstring(x, sep=" ") for x in user_embeddings]).reshape([user_count, self.dimensions])
		
		print("Normalizing users.. ", end='', flush=True)	
		user_embeddings = np.apply_along_axis(lambda e: self.normalize_array(e), 0, user_embeddings)
		return user_ids, user_embeddings










