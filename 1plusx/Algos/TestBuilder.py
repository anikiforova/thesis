
import numpy as np

from Metadata import Metadata
from TestMetadata import TestMetadata
from TargetSplitType import TargetSplitType
from SimulationType import SimulationType

def build_gp_test(meta, click_percent = 0.2, kernel = "Matern", nu = 1.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2):
	t = TestMetadata(meta)
	t.gp_running_algo	= True
	t.click_percent		= click_percent
	t.kernel_name		= kernel
	t.nu 				= nu
	t.length_scale		= length_scale
	t.cluster_count		= cluster_count
	t.alpha				= alpha
	t.train_part 			= rec_part
	t.recommendation_part 	= rec_part
	t.hours 				= h

	return t

def build_nn_test(meta, click_percent = 0.2, learning_rate = 0.001, h = 12, rec_part = 0.2):
	t = TestMetadata(meta)
	t.nn_running_algo	= True
	t.learning_rate		= learning_rate
	t.click_percent		= click_percent
	t.train_part 			= rec_part
	t.recommendation_part 	= rec_part
	t.hours 				= h

	return t

def build_lin_test(meta, click_percent = 0.2, alpha = 0.1, h = 12, rec_part = 0.2):
	t = TestMetadata(meta)
	t.alpha				= alpha
	t.click_percent		= click_percent
	t.train_part 			= rec_part
	t.recommendation_part 	= rec_part
	t.hours 				= h

	return t

def get_lin_tests(meta, hours = 12):
	tests = list()
	for alpha in [0.1, 0.01, 0.001]: # 0.1, 
		#for rec_part in [0.02, 0.05, 0.1, 0.2, 0.5]:
		for rec_part in [0.2]:
			#for click_percent in [0.0, 0.2, 0.5]:
			for click_percent in [0.0]:
				tests.append(build_lin_test(meta, click_percent = click_percent, alpha = alpha, h = hours, rec_part = rec_part))
	return tests

def get_lin_tests_50(meta, hours = 12):
	tests = list()
	rec_part = 0.033
	click_percent = 0.0

	for alpha in [0.1, 0.001]: # 0.1, 
		tests.append(build_lin_test(meta, click_percent = click_percent, alpha = alpha, h = hours, rec_part = rec_part))
	return tests


def get_lin_tests_mini(meta, hours = 12):
	tests = list()
	click_percent 	= 0.0
	for rec_part in [0.2]:
		for alpha in [0.001]:
			tests.append(build_lin_test(meta, click_percent = click_percent, alpha = alpha, h = hours, rec_part = rec_part))
	return tests

def get_lin_alpha_tests(meta, hours = 12):
	tests = list()
	for alpha in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
		tests.append(build_lin_test(meta, click_percent = 0.0, alpha = alpha, h = hours, rec_part = 0.2))
	return tests

def get_lin_test(meta, hours = 12):
	tests = list()
	tests.append(build_lin_test(meta, click_percent = 0.0, alpha = 0.001, h = hours, rec_part = 0.033))
	return tests

def build_target_test(meta, alpha, hours, target_percent, target_split, target_alpha, early_update):
	test = build_lin_test(meta, click_percent = 0.0, alpha = alpha, h = hours, rec_part = 0.0)
	test.target_percent = target_percent
	test.target_algo = True
	test.target_split = target_split
	test.target_alpha = target_alpha
	test.early_update = early_update
	return test

def get_lin_multi_test(meta, hours = 12):
	tests = list()
	for alpha in [0.1]: #[0.1, 0.01, 0.001]:
		test = build_lin_test(meta, click_percent = 0.0, alpha = alpha, h = hours, rec_part = 0.033)
		test.normalize_ctr = False
		tests.append(test)
			
	return tests

def get_simulation_lin_multi_test(meta, hours = 12):
	tests = list()
	for simulation_type in [SimulationType.LOWER, SimulationType.HINDSIGHT]:
		for alpha in [0.1]: #[0.1, 0.01, 0.001]:
			test = build_lin_test(meta, click_percent = 0.0, alpha = alpha, h = hours, rec_part = 0.0)
			test.normalize_ctr = False
			test.is_simulation = True
			test.simulation_type = simulation_type
			tests.append(test)

	return tests

def get_simulation_hindsight_lin_multi_test(meta, hours = 12):
	tests = list()
	chisquare_dfs = [0, 1, 5, 10, 20, 100]
	simulation_indexes = [2, 5, 5, 5, 5, 5] 

	for alpha in [0.1, 0.001]:
		for chi_df, simulation_index in zip(chisquare_dfs, simulation_indexes):
			test = build_lin_test(meta, click_percent = 0.0, alpha = alpha, h = hours, rec_part = 0.0)
			test.normalize_ctr = True
			test.is_simulation = True
			test.simulation_type = SimulationType.HINDSIGHT
			test.simulation_index = simulation_index
			test.chi_df = chi_df
			tests.append(test)

	return tests

def get_simulation_hindsight_lin_multi_target_test_chi(meta, hours = 12):
	tests = list()
	chisquare_dfs = [1, 5, 10, 20, 100]
	simulation_indexes = [6, 6, 6, 6, 6] 
	target_percent = 0.8
	
	for chi_df, simulation_index in zip(chisquare_dfs, simulation_indexes):
		for alpha in [0.1]: #, 0.001]:
			test = build_target_test(meta, alpha = alpha, hours = hours, target_percent = target_percent, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = True)
			test.normalize_ctr = True
			test.normalize_target_value = True
			test.is_simulation = True
			test.simulation_type = SimulationType.HINDSIGHT
			test.simulation_index = simulation_index
			test.chi_df = chi_df
			tests.append(test)

	return tests

def get_simulations_multi_test(meta, hours = 12):
	tests = list()
	
	alpha = 0.1

	chisquare_df = 1
	chi_alphas = [1, 4]
	for simulation_index in [5]:
		for chi_alpha in chi_alphas:
			test = build_lin_test(meta, click_percent = 0.0, alpha = alpha, h = hours, rec_part = 0.0)
			test.normalize_ctr = False
			test.is_simulation = True
			test.simulation_type = SimulationType.HINDSIGHT
			test.simulation_index = simulation_index
			test.chi_df = chisquare_df
			test.chi_alpha = chi_alpha
			tests.append(test)

	return tests

def get_simulation_lin_multi_target_test(meta, hours = 12):
	tests = list()
	chi_dfs = [1, 10, 100]
	chi_alphas = [2, 4]
	
	alpha = 0.1

	simulation_index = 5
	target_percent = 0.8
	
	for chi_alpha in chi_alphas:
		for chi_df in chi_dfs:
			test = build_target_test(meta, alpha = alpha, hours = hours, target_percent = target_percent, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = True)
			test.normalize_ctr = True
			test.normalize_target_value = True
			test.is_simulation = True
			test.simulation_type = SimulationType.HINDSIGHT
			test.simulation_index = 5
			test.chi_df = chi_df
			test.chi_alpha = chi_alpha
			tests.append(test)

	return tests

def get_simulation_lower_lin_multi_target_test(meta, hours = 12):
	tests = list()
	for simulation_type in [SimulationType.LOWER]:
		for target_percent in [0.6, 0.8, 1]:
			for alpha in [0.1, 0.0001]:
				test = build_target_test(meta, alpha = alpha, hours = hours, target_percent = target_percent, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = True)
				test.normalize_ctr = True
				test.is_simulation = True
				test.simulation_type = simulation_type
				tests.append(test)

	return tests

def get_simulation_hindsight_lin_multi_target_test(meta, hours = 12):
	tests = list()
	for simulation_type in [SimulationType.HINDSIGHT]:
		for target_percent in [1]:
			for alpha in [0.1, 0.0001]:
				test = build_target_test(meta, alpha = alpha, hours = hours, target_percent = target_percent, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = True)
				test.normalize_ctr = True
				test.is_simulation = True
				test.simulation_type = simulation_type
				tests.append(test)

	return tests

def full_target_tests(meta, hours = 12):
	tests = list()
	for target_percent in [0.6, 0.8, 1]:
		for early_update in [True, False]:
			for target_split in [TargetSplitType.DAILY, TargetSplitType.NO_SPLIT]:
				test = build_target_test(meta, 
					alpha = 0.1, 
					hours = hours, 
					target_percent = target_percent, 
					target_split = target_split, 
					target_alpha = 1, 
					early_update = early_update)
				test.normalize_target_value = True	
				tests.append(test)

	for crop_percent in [0.1 ,0.2, 0.4]:
		test = build_target_test(meta, alpha = 0.1, hours = hours, target_percent = 1, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = False)
		test.crop_minimal_target = True
		test.crop_percent = crop_percent
		test.normalize_target_value = True	
		tests.append(test)

	for early_update in [True, False]:
		test = build_target_test(meta, alpha = 0.1, hours = hours, target_percent = 1, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = early_update)
		test.normalize_target_value = True	
		tests.append(test)

	return tests

def baseline_test(meta, hours = 12):
	tests = list()

	test = build_target_test(meta, 
		alpha = 0.1, 
		hours = hours, 
		target_percent = 1, 
		target_split = TargetSplitType.NO_SPLIT, 
		target_alpha = 1, 
		early_update = False)
	test.normalize_target_value = False	

	tests.append(test)

	return tests;

def basic_feature_target_tests1(meta, hours = 12):
	tests = list()
	
	for early_update in [True]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = TargetSplitType.NO_SPLIT, 
			target_alpha = 1, 
			early_update = early_update)
		test.normalize_target_value = False	
		tests.append(test)

	for target_split in [TargetSplitType.DAILY]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = target_split, 
			target_alpha = 1, 
			early_update = False)
		test.normalize_target_value = False	
		tests.append(test)

	for normalize in [True]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = TargetSplitType.NO_SPLIT, 
			target_alpha = 1, 
			early_update = False)
		test.normalize_target_value = normalize	
		tests.append(test)

	for target_percent in [0.6, 0.8]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = target_percent, 
			target_split = TargetSplitType.NO_SPLIT, 
			target_alpha = 1, 
			early_update = False)
		test.normalize_target_value = False	
		tests.append(test)
	return tests

def basic_feature_target_tests2(meta, hours = 12):
	tests = list()
	
	for target_percent in [0.4]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = target_percent, 
			target_split = TargetSplitType.NO_SPLIT, 
			target_alpha = 1, 
			early_update = False)
		test.normalize_target_value = False	
		tests.append(test)

	for crop_percent in [0.1 ,0.2, 0.4]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = TargetSplitType.NO_SPLIT, 
			target_alpha = 1, 
			early_update = False)
		test.crop_minimal_target = True
		test.crop_percent = crop_percent
		test.normalize_target_value = False	
		tests.append(test)
	return tests

def full_feature_target_tests(meta, hours = 12):
	tests = list()
	for target_percent in [0.6, 0.8, 1]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = target_percent, 
			target_split = TargetSplitType.DAILY, 
			target_alpha = 1, 
			early_update = True)
		test.normalize_target_value = True	
		tests.append(test)

	for crop_percent in [0.1 ,0.2, 0.4]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = TargetSplitType.DAILY, 
			target_alpha = 1, 
			early_update = False)
		test.crop_minimal_target = True
		test.crop_percent = crop_percent
		test.normalize_target_value = True	
		tests.append(test)

	for early_update in [True, False]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = TargetSplitType.DAILY, 
			target_alpha = 1, 
			early_update = early_update)
		test.normalize_target_value = True	
		tests.append(test)

	for target_split in [TargetSplitType.DAILY, TargetSplitType.NO_SPLIT]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = target_split, 
			target_alpha = 1, 
			early_update = True)
		test.normalize_target_value = True	
		tests.append(test)

	for normalize in [True, False]:
		test = build_target_test(meta, 
			alpha = 0.1, 
			hours = hours, 
			target_percent = 1, 
			target_split = TargetSplitType.DAILY, 
			target_alpha = 1, 
			early_update = True)
		test.normalize_target_value = normalize	
		tests.append(test)

	return tests

def get_multi_target_normalization_tests(meta, hours = 12):
	tests = list()

	# for crop_percent in [0.5, 0.1 ,0.2]:
	# 	test = build_target_test(meta, alpha = 0.1, hours = hours, target_percent = 1, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = False)
	# 	test.crop_minimal_target = True
	# 	test.crop_percent = crop_percent

	# 	tests.append(test)

	test = build_target_test(meta, alpha = 0.1, hours = hours, target_percent = 1, target_split = TargetSplitType.DAILY, target_alpha = 1, early_update = True)
	test.normalize_target_value = True	
	tests.append(test)

	return tests

def get_lin_multi_target_test(meta, hours = 12):
	tests = list()

	for alpha in [0.1, 0.001]:
		for early_update in [True, False]:
			for target_split in [TargetSplitType.DAILY, TargetSplitType.NO_SPLIT]:
				tests.append(build_target_test(meta, alpha = alpha, hours = hours, target_percent = 1, target_split = target_split, target_alpha = 1, early_update = early_update))	
			
	return tests

def get_lin_multi_mix_test(meta, hours = 12):
	tests = list()
	for alpha in [0.1, 0.01, 0.001]: 
		test = build_lin_test(meta, click_percent = 0.0, alpha = alpha, h = hours, rec_part = 0.2)
		test.normalize_ctr = False
		tests.append(test)
			
	return tests

def get_lin_multi_target_mix_test(meta, hours = 12):
	tests = list()
	for target_split in [TargetSplitType.NO_SPLIT]:
		for target_alpha in [1]: 
			for alpha in [0.1]:
				for target_percent in [1]:
					test = build_target_test(meta, alpha = alpha, hours = hours, target_percent = target_percent, target_split = target_split, target_alpha = target_alpha)
					test.recommendation_part 	= 0.2
					test.train_part 			= 0.2
					tests.append(test)
				
	return tests

def get_lin_multi_target_test_mini(meta, hours = 12):
	tests = list()
	alpha = 0.1
	target_alpha = 1
	target_percent = 1
	early_update = True
	for target_split in [TargetSplitType.DAILY, TargetSplitType.NO_SPLIT]:	
		tests.append(build_target_test(meta, alpha = alpha, hours = hours, target_percent = target_percent, target_split = target_split, target_alpha = target_alpha, early_update = early_update))
				
	return tests

def get_simulation_lin_multi_test(meta, hours = 12):
	tests = list()
	alpha = 0.1
	target_alpha = 1
	target_percent = 1
	early_update = True

	for simulation_type in [SimulationType.LOWER, SimulationType.HINDSIGHT]:
		for target_split in [TargetSplitType.DAILY, TargetSplitType.NO_SPLIT]:	
			test = build_target_test(meta, alpha = alpha, hours = hours, target_percent = target_percent, target_split = target_split, target_alpha = target_alpha, early_update = early_update)
			test.is_simulation = True
			test.simulation_type = simulation_type
			tests.append(test)

	return tests

# DONE
def get_random_multi_tests(meta, hours = 12):
	t = TestMetadata(meta)
	t.recommendation_part 	= 0.0
	t.hours 				= hours
	t.click_percent			= 0.0

	return [t]

def get_random_tests(meta, hours = 12):
	t = TestMetadata(meta)
	t.recommendation_part 	= 0.2
	t.hours 				= hours
	t.click_percent			= 0.0

	return [t]

def get_random_multi_simulation_tests(meta, hours):
	t = TestMetadata(meta)
	t.recommendation_part 	= 0.0
	t.train_part 			= 0.0
	t.hours 				= hours
	t.click_percent			= 0.0
	t.is_simulation 		= True
	t.simulation_type		= SimulationType.HINDSIGHT
	t.target_percent		= 0.8
	
	return [t]

def get_nn_tests(meta):
	return [
	# test learning rate
 	build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, click_percent = 0.2), 
 	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.2, click_percent = 0.2),
 #	build_nn_test(meta, learning_rate = 0.0001, h = 12, rec_part = 0.2, click_percent = 0.2),
	
# # 	# test train part
 	# build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.1, click_percent = 0.2),
	# build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.2, click_percent = 0.2), # duplicate
	# build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.5, click_percent = 0.2),

	# build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.1, click_percent = 0.2),
	# build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.1, click_percent = 0.5),
	# build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.2, click_percent = 0.2), # duplicate
	# build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.5, click_percent = 0.2),
	# build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.5, click_percent = 0.5),
	
	# # click percent
#	build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, click_percent = 0.5), 
#	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.2, click_percent = 0.5),
#	build_nn_test(meta, learning_rate = 0.0001, h = 12, rec_part = 0.2, click_percent = 0.5),
	#build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, click_percent = 0.0) 
		]

# Run uncommented tests.
def get_gp_tests(meta):
	return [	
# test no eq of clicks
#build_gp_test(meta, click_percent = 0.0, nu = 1.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0.0, nu = 2.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
#build_gp_test(meta, click_percent = 0.0, nu = 3.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
# test eq clicks with 50% 
#build_gp_test(meta, click_percent = 0.5, nu = 1.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
#build_gp_test(meta, click_percent = 0.5, nu = 2.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
#build_gp_test(meta, click_percent = 0.5, nu = 3.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
#try behavior with 0.2 click percent and different nus
#build_gp_test(meta, click_percent = 0.2, nu = 1.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
#build_gp_test(meta, click_percent = 0.2, nu = 2.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
#build_gp_test(meta, click_percent = 0.2, nu = 3.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
# try different length scale
#build_gp_test(meta, click_percent = 0.2, nu = 1.5, length_scale = 200, cluster_count = 10, h = 12, rec_part = 0.2),
#build_gp_test(meta, click_percent = 0.2, nu = 2.5, length_scale = 200, cluster_count = 10, h = 12, rec_part = 0.2),
#try different timespan
#build_gp_test(meta, click_percent = 0.2, nu = 1.5, length_scale = 100, cluster_count = 10, h = 4,  rec_part = 0.2),
#build_gp_test(meta, click_percent = 0.2, nu = 2.5, length_scale = 100, cluster_count = 10, h = 4,  rec_part = 0.2),
# # try different cluster count
# build_gp_test(meta, click_percent = 0.2, nu = 1.5, length_scale = 100, cluster_count = 20, h = 12, rec_part = 0.2),
# build_gp_test(meta, click_percent = 0.2, nu = 2.5, length_scale = 100, cluster_count = 20, h = 12, rec_part = 0.2)
# build_gp_test(meta, click_percent = 0.0, nu = 1.5, length_scale = 100, cluster_count = 20, h = 12, rec_part = 0.2),
# build_gp_test(meta, click_percent = 0.0, nu = 2.5, length_scale = 100, cluster_count = 20, h = 12, rec_part = 0.2),
# build_gp_test(meta, click_percent = 0.0, nu = 3.5, length_scale = 100, cluster_count = 20, h = 12, rec_part = 0.2),

# build_gp_test(meta, click_percent = 0.0, nu = 1.5, length_scale = 100, cluster_count = 20, h = 4, rec_part = 0.2),
# build_gp_test(meta, click_percent = 0.0, nu = 2.5, length_scale = 100, cluster_count = 20, h = 4, rec_part = 0.2),
# build_gp_test(meta, click_percent = 0.0, nu = 3.5, length_scale = 100, cluster_count = 20, h = 4, rec_part = 0.2),

# build_gp_test(meta, click_percent = 0.0, nu = 1.5, length_scale = 200, cluster_count = 20, h = 12, rec_part = 0.2),
# build_gp_test(meta, click_percent = 0.0, nu = 2.5, length_scale = 200, cluster_count = 20, h = 12, rec_part = 0.2),
# build_gp_test(meta, click_percent = 0.0, nu = 3.5, length_scale = 200, cluster_count = 20, h = 12, rec_part = 0.2)

			]