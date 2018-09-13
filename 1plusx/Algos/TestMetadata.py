import numpy as np
from pandas import read_csv
from pandas import DataFrame

from TargetSplitType import TargetSplitType

class TestMetadata:

	click_percent					= 0.2


	# GP
	gp_running_algo					= False
	kernel_name						= ""
	noiseVar 						= 0.01
	nu 								= 1.5
	length_scale					= 100
	cluster_count	 				= 10
	
	# NN
	nn_running_algo					= False
	learning_rate 					= 0.001
	hidden_layers					= 10

	# LinUCB, TS_Lin, GP (only as scale)
	alpha							= 0.0

	hours 							= 12

	train_part 						= 0.2
	recommendation_part 			= 0.2

	early_update					= False
	target_algo						= False
	target_percent					= 0.8
	target_split					= TargetSplitType.NO_SPLIT
	target_alpha					= 1
	normalize_ctr					= True

	def __init__(self, meta):
		self.meta 			= meta
		self.length_scale	= self.meta.dimensions

	def get_time_between_updates_in_seconds(self):
		return 60 * 60 * self.hours

	def get_algo_info(self):
		info = "{},H:{},Alpha:{},ClickPercent:{:.2}".format(self.meta.algo_name, self.hours, self.alpha, self.click_percent)

		if self.target_algo:
			info = "{},TargetPercent:{},TargetSplit:{},TargetAlpha:{},EarlyUpdate:{}".format(info, self.target_percent, self.target_split.name, self.target_alpha, self.early_update)
		
		info = "{},Train:{},Rec:{}".format(info, self.train_part, self.recommendation_part)

		if self.gp_running_algo:
			info =  "{},#Clusters:{},LengthScale:{},Nu:{:.2},Kernel:{}".format(info, self.cluster_count, self.length_scale, self.nu, self.kernel_name)

		if self.nn_running_algo:
			info = "{},LearningRate:{},HiddenLayers:{}".format(info, self.learning_rate, self.hidden_layers)
				
		return info

	def get_algo_column_info(self):
		info = "{},{},{},{}".format(self.meta.algo_name, self.hours, self.alpha, self.click_percent)
		
		if self.target_algo:
			info = "{},{},{},{},{}".format(info, self.target_percent, self.target_split.name, self.target_alpha, self.early_update)
		

		info = "{},{},{}".format(info, self.train_part, self.recommendation_part)

		if self.gp_running_algo:
			info =  "{},{},{},{:.2},{}".format(info, self.cluster_count, self.length_scale, self.nu, self.kernel_name)

		if self.nn_running_algo:
			info = "{},{},{}".format(info, self.learning_rate, self.hidden_layers)

		return info

	def get_algo_column_names(self):
		info = "Method,Hours,Alpha,EqClicks"

		if self.target_algo:
			info = info + ",TargetPercent,TargetSplit,TargetAlpha,EarlyUpdate"
		
		info = info + ",TrainPart,RecommendationPart"

		if self.gp_running_algo:
			info =  info + ",ClusterCount,LengthScale,Nu,Kernel"

		if self.nn_running_algo:
			info = info + ",LearningRate,HiddenLayers"

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










