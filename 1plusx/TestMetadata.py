import numpy as np
from pandas import read_csv
from pandas import DataFrame

class TestMetadata:

	equalize_clicks 			= True
	no_click_percent 			= 0.8
	click_percent				= 1 - no_click_percent
	
	# GP
	gp_running_algo				= True
	kernel_name					= ""
	noiseVar 					= 0.01
	nu 							= 1.5
	length_scale				= 100
	cluster_count	 			= 10
	
	alpha						= 0.0

	hours 							= 12
	time_between_updates_in_seconds = 60 * 60 * hours # 1 hour

	def __init__(self, meta):
		self.meta 			= meta
		self.length_scale	= self.meta.dimensions

	def get_additional_info(self):
		info = "H:{}".format(self.hours)

		if self.equalize_clicks:
			info =  "{},Eq:{}".format(info, self.click_percent)

		if self.gp_running_algo:
			info =  "{},C:{},L:{},NU:{},K:{}".format(info, self.cluster_count, self.length_scale, self.nu, self.kernel_name)

		return info

	def normalize_array(self, array):
		min_value = np.min(array)
		max_value = np.max(array)
		return (array - min_value)/(max_value - min_value)

	def set_local_params(self, alpha, user_recommendation_part, user_train_part):
		self.alpha = alpha
		self.user_recommendation_part = user_recommendation_part
		self.user_train_part = user_train_part
		
	def read_cluster_embeddings(self):
		print("Reading and normalizing cluster data.. ", end='', flush=True)
		clusters = read_csv("{0}//all_users_clusters_{1}.csv".format(self.meta.path, self.cluster_count), header=0)

		cluster_embeddings = np.array(clusters["Cluster"])
		cluster_embeddings = np.array([np.fromstring(x, sep=" ") for x in cluster_embeddings]).reshape([self.cluster_count, self.meta.dimensions])
		cluster_embeddings 	= np.apply_along_axis(lambda e: self.normalize_array(e), 0, cluster_embeddings)

		print(" Done.")

		return cluster_embeddings

	def read_user_assignments(self):
		user_assignments = read_csv("{0}//all_users_cluster_assignment_{1}.csv".format(self.meta.path, self.cluster_count), header=0)
		return np.array(user_assignments["ClusterId"].values)

	# def save_user_clusters(self, kernels):
	# 	print("Saving kernels.. ", end='', flush=True)	
	# 	output = open("{0}//all_users_kernels_{1}_2.csv".format(self.path, self.cluster_count), "w")
	# 	output.write("Kernels\n")
	# 	for k in kernels:
	# 		user_str = np.array2string(k, separator=' ')[1:-1].replace('\n', '')
	# 		output.write("{}\n".format(user_str))
			
	# 	output.close()
	# 	print(" Done.")

	# def read_user_clusters(self, user_count):
	# 	print("Reading kernels.. ", end='', flush=True)	
	# 	kernels = read_csv("{0}//all_users_kernels_{1}.csv".format(self.path, self.cluster_count), header=0)
	# 	print(" Done.")

	# 	print("Parsing kernels.. ", end='', flush=True)	
	# 	kernels = np.array(kernels["Kernels"])
	# 	kernels = np.array([np.fromstring(x, sep=" ") for x in kernels])
	# 	kernels = [item for sublist in kernels for item in sublist]
	# 	kernels = np.array(kernels).reshape([user_count, self.cluster_count])
	# 	print(" Done.")

	# 	return kernels









