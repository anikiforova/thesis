
import numpy as np

from Metadata import Metadata
from TestMetadata import TestMetadata

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

def get_lin_tests(meta):
	tests = list()
	for alpha in [0.01, 0.001]: # 0.1, 
		for rec_part in [0.02, 0.05, 0.1, 0.2, 0.5]:
			for click_percent in [0.0, 0.2, 0.5]:
				tests.append(build_lin_test(meta, click_percent = click_percent, alpha = alpha, h = 12, rec_part = rec_part))
	return tests

def get_lin_tests_mini(meta, hours = 12):
	tests = list()
	click_percent 	= 0.0
	for rec_part in [0.02, 0.05, 0.1, 0.2, 0.5]:
		for alpha in [0.1, 0.01, 0.001]:
			tests.append(build_lin_test(meta, click_percent = click_percent, alpha = alpha, h = hours, rec_part = rec_part))
	return tests

# DONE
def get_random_tests(meta):
	t = TestMetadata(meta)
	t.recommendation_part 	= 0.2
	t.hours 				= 12

	return [t]

def get_nn_tests(meta):
	return [
	# test learning rate
	build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, click_percent = 0.2), 
	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.2, click_percent = 0.2),
	build_nn_test(meta, learning_rate = 0.0001, h = 12, rec_part = 0.2, click_percent = 0.2),
	
# 	# test train part
	build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.1, click_percent = 0.2),
	# build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.2, click_percent = 0.2), # duplicate
	build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.5, click_percent = 0.2),

	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.1, click_percent = 0.2),
	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.1, click_percent = 0.5),
	# build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.2, click_percent = 0.2), # duplicate
	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.5, click_percent = 0.2),
	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.5, click_percent = 0.5),
	
	# click percent
	build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, click_percent = 0.5), 
	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.2, click_percent = 0.5),
	build_nn_test(meta, learning_rate = 0.0001, h = 12, rec_part = 0.2, click_percent = 0.5),
	build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, click_percent = 0.0) 
		]

# Run uncommented tests.
def get_gp_tests(meta):
	return [	
# test no eq of clicks
build_gp_test(meta, click_percent = 0, nu = 1.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0, nu = 2.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0, nu = 3.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
# test eq clicks with 50% 
build_gp_test(meta, click_percent = 0.5, nu = 1.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0.5, nu = 2.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0.5, nu = 3.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
#try behavior with 0.2 click percent and different nus
build_gp_test(meta, click_percent = 0.2, nu = 1.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0.2, nu = 2.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0.2, nu = 3.5, length_scale = 100, cluster_count = 10, h = 12, rec_part = 0.2),
# try different length scale
build_gp_test(meta, click_percent = 0.2, nu = 1.5, length_scale = 200, cluster_count = 10, h = 12, rec_part = 0.2),
build_gp_test(meta, click_percent = 0.2, nu = 2.5, length_scale = 200, cluster_count = 10, h = 12, rec_part = 0.2),
#try different timespan
build_gp_test(meta, click_percent = 0.2, nu = 1.5, length_scale = 100, cluster_count = 10, h = 4,  rec_part = 0.2),
build_gp_test(meta, click_percent = 0.2, nu = 2.5, length_scale = 100, cluster_count = 10, h = 4,  rec_part = 0.2),
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