import numpy as np
from pandas import read_csv
from pandas import DataFrame

class Metadata:
	soft_click 					= False
	filter_clickers 			= False
	remove_duplicate_no_clicks 	= False

	dimensions 					= 100
	initialize_user_embeddings  = True
	algo_name = ""

	base_path = "../../RawData/Campaigns/"

	total_lines = dict({
		0:0, 
		5:dict({"2018-08-15": 3176464,
				"2018-08-16": 2549790,
				"2018-08-17": 2834130,
				"2018-08-18": 2770644,
				"2018-08-19": 2315645,
				"2018-08-20": 2135887,
				"2018-08-21": 3058886,
				"2018-08-22": 2166106,
				"2018-08-23": 1997451,
				"2018-08-24": 1451283,
				"2018-08-25": 1574728,
				"2018-08-26": 1431289,
				"2018-08-27": 1328424,
				"2018-08-28": 1648197}),
		10:0,
		597165:2257957, 
		837817:5409623, 
		722100:8475542, 
		809153:14392075,
		847460:3937517,
		856805:12806271,
		858140:4517501,
		865041:4357812,
		866128:4819814})


	def __init__(self, algo_name, campaign_id, initialize_user_embeddings = True):
		self.campaign_id = campaign_id
		self.algo_name = algo_name
		self.initialize_user_embeddings = initialize_user_embeddings
		self.path = "{0}/{1}/Processed".format(self.base_path, self.campaign_id)

	def normalize_array(array):
		min_value = np.min(array)
		max_value = np.max(array)
		return (array - min_value)/(max_value - min_value)

	def read_user_embeddings(self, embeddings_file_name_postfix	= ""):
		print("Reading users.. ", end='', flush=True)
		filePath = "{0}/all_users{1}.csv".format(self.path, embeddings_file_name_postfix)
		return Metadata.read_user_embeddings_by_path(filePath)

	def read_user_embeddings_by_path(filePath):
		print("({})".format(filePath))
		users = read_csv(filePath, header=0)
		
		print("Parsing users.. ", end='', flush=True)	
		user_ids = np.array(users["UserHash"])
		user_count = len(user_ids)
		user_embeddings = np.array(users["UserEmbedding"])
		user_embeddings = np.array([np.fromstring(x, sep=" ") for x in user_embeddings]).reshape([user_count, Metadata.dimensions])
		
		print("Normalizing users.. ", end='', flush=True)	
		user_embeddings = np.apply_along_axis(lambda e: Metadata.normalize_array(e), 0, user_embeddings)
		return user_ids, user_embeddings










