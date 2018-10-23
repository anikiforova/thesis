import numpy as np
from pandas import read_csv
from pandas import DataFrame

from TargetSplitType import TargetSplitType
from SimulationType import SimulationType

from SimulationType import get_friendly_name

class TestMetadata:

	click_percent					= 0.2

	# Simulation meta
	is_simulation					= False
	simulation_type					= SimulationType.HINDSIGHT
	simulation_index				= 0
	chi_df							= 0
	chi_alpha						= 1

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

	# Target Budget test meta
	early_update					= False
	target_algo						= False
	target_percent					= 0.8
	target_split					= TargetSplitType.NO_SPLIT
	target_alpha					= 1
	normalize_ctr					= True
	normalize_target_value			= False
	crop_minimal_target				= False
	crop_percent					= 0.0

	def __init__(self, meta):
		self.meta 			= meta
		self.length_scale	= self.meta.dimensions

	def get_time_between_updates_in_seconds(self):
		return 60 * 60 * self.hours

	def get_algo_info(self):
		info = "{},H:{},Alpha:{},ClickPercent:{:.2}".format(self.meta.algo_name, self.hours, self.alpha, self.click_percent)

		if self.target_algo:
			info = "{},TargetPercent:{},TargetSplit:{},TargetAlpha:{},EarlyUpdate:{},CropPercent:{},NormalizeTargetValue:{}".format(info, self.target_percent, self.target_split.name, self.target_alpha, self.early_update, self.crop_percent, self.normalize_target_value)
		
		info = "{},Train:{},Rec:{}".format(info, self.train_part, self.recommendation_part)

		if self.gp_running_algo:
			info =  "{},#Clusters:{},LengthScale:{},Nu:{:.2},Kernel:{}".format(info, self.cluster_count, self.length_scale, self.nu, self.kernel_name)

		if self.nn_running_algo:
			info = "{},LearningRate:{},HiddenLayers:{}".format(info, self.learning_rate, self.hidden_layers)
			
		if self.is_simulation:
			info = "{},SimulationType:{}".format(info, get_friendly_name(self.simulation_type))
			if self.simulation_index != 0:
				info = "{},SimulationIndex:{},ChiDF:{},ChiAlpha:{}".format(info, self.simulation_index, self.chi_df, self.chi_alpha)

		return info

	def get_algo_column_info(self):
		info = "{},{},{},{}".format(self.meta.algo_name, self.hours, self.alpha, self.click_percent)
		
		if self.target_algo:
			info = "{},{},{},{},{},{},{}".format(info, self.target_percent, self.target_split.name, self.target_alpha, self.early_update, self.crop_percent, self.normalize_target_value)
		

		info = "{},{},{}".format(info, self.train_part, self.recommendation_part)

		if self.gp_running_algo:
			info =  "{},{},{},{:.2},{}".format(info, self.cluster_count, self.length_scale, self.nu, self.kernel_name)

		if self.nn_running_algo:
			info = "{},{},{}".format(info, self.learning_rate, self.hidden_layers)

		if self.is_simulation:
			info = "{},{}".format(info, get_friendly_name(self.simulation_type))
			if self.simulation_index != 0:
				info = "{},{},{},{}".format(info, self.simulation_index, self.chi_df, self.chi_alpha)
			
		return info

	def get_algo_column_names(self):
		info = "Method,Hours,Alpha,EqClicks"

		if self.target_algo:
			info = info + ",TargetPercent,TargetSplit,TargetAlpha,EarlyUpdate,CropPercent,NormalizeTargetValue"
		
		info = info + ",TrainPart,RecommendationPart"

		if self.gp_running_algo:
			info =  info + ",ClusterCount,LengthScale,Nu,Kernel"

		if self.nn_running_algo:
			info = info + ",LearningRate,HiddenLayers"

		if self.is_simulation:
			info = info + ",SimulationType"
			if self.simulation_index != 0:
				info = info + ",SimulationIndex,ChiDF,ChiAlpha"
			
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










