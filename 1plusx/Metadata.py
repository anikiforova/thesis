import numpy as np
from pandas import read_csv
from pandas import DataFrame

class Metadata:
	soft_click 					= False
	filter_clickers 			= False
	remove_duplicate_no_clicks 	= False

	equalize_clicks 			= False
	no_click_percent 			= 0.80
	click_percent				= 1 - no_click_percent
	
	dimensions 					= 100
	cluster_count	 			= 10
	cluster_name_postfix 		= ""

	calculate_users_kernels 	= False
	# GP
	noiseVar 					= 0.01

	alpha						= 0.0
	user_recommendation_part 	= 0.02
	user_train_part 			= 0.02

	hours 							= 1
	time_between_updates_in_seconds = 60 * 60 * hours # 1 hour

	total_lines = 14392074

	path = "..//..//RawData//Campaigns//809153//Processed"

	def __init__(self):
		pass

	def normalize_array(self, array):
		min_value = np.min(array)
		max_value = np.max(array)
		return (array - min_value)/(max_value - min_value)

	def set_local_params(self, alpha, user_recommendation_part, user_train_part):
		self.alpha = alpha
		self.user_recommendation_part = user_recommendation_part
		self.user_train_part = user_train_part

	def construct_algo_name(self, algo):
		algo_name = algo + self.cluster_name_postfix

		if self.equalize_clicks:
			algo_name += "_E_" + str(self.no_click_percent) 

		return algo_name

	def read_cluster_embeddings(self):
		print("Reading and normalizing cluster data.. ", end='', flush=True)
		clusters = read_csv("{0}//all_users_clusters_{1}.csv".format(self.path, self.cluster_count), header=0)

		cluster_embeddings = np.array(clusters["Cluster"])
		cluster_embeddings = np.array([np.fromstring(x, sep=" ") for x in cluster_embeddings]).reshape([self.cluster_count, self.dimensions])
		cluster_embeddings 	= np.apply_along_axis(lambda e: self.normalize_array(e), 0, cluster_embeddings)

		# clustered_users = read_csv("{0}//all_users_clustered_{1}.csv".format(path, cluster_count), header=0)
		# clustered_user_embeddings = np.array(clustered_users["UserEmbedding"])
		# clustered_user_embeddings = np.array([np.fromstring(x, sep=" ") for x in clustered_user_embeddings]).reshape([len(user_ids), cluster_count])
		# clustered_user_embeddings = np.apply_along_axis(lambda e: self.normalize_array(e), 0, clustered_user_embeddings)
		print(" Done.")

		return cluster_embeddings

	def read_user_assignments(self):
		user_assignments = read_csv("{0}//all_users_cluster_assignment_{1}.csv".format(self.path, self.cluster_count), header=0)
		return np.array(user_assignments["ClusterId"].values)

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

	def save_user_clusters(self, kernels):
		print("Saving kernels.. ", end='', flush=True)	
		output = open("{0}//all_users_kernels_{1}.csv".format(self.path, self.cluster_count), "w")
		output.write("Kernels\n")
		for k in kernels:
			user_str = np.array2string(k, separator=' ')[1:-1].replace('\n', '')
			output.write("{}\n".format(user_str))
			
		output.close()
		print(" Done.")

	def read_user_clusters(self, user_count):
		print("Reading kernels.. ", end='', flush=True)	
		kernels = read_csv("{0}//all_users_kernels_{1}.csv".format(self.path, self.cluster_count), header=0)
		print(" Done.")

		print("Parsing kernels.. ", end='', flush=True)	
		kernels = np.array(kernels["Kernels"])
		kernels = np.array([np.fromstring(x, sep=" ") for x in kernels])
		kernels = [item for sublist in kernels for item in sublist]
		kernels = np.array(kernels).reshape([user_count, self.cluster_count])
		print(" Done.")

		return kernels










