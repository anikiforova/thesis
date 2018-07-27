
import numpy as np

from Metadata import Metadata
from TestMetadata import TestMetadata

def build_gp_test(meta, eq = True, click_percent = 0.2, kernel = "Matern", nu = 1.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2):
	t = TestMetadata(meta)
	t.gp_running_algo	= True
	t.equalize_clicks 	= eq
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

def build_nn_test(meta, eq = True, click_percent = 0.2, learning_rate = 0.001, h = 12, rec_part = 0.2):
	t = TestMetadata(meta)
	t.nn_running_algo	= True
	t.learning_rate		= learning_rate
	t.equalize_clicks 	= eq
	t.click_percent		= click_percent
	t.train_part 			= rec_part
	t.recommendation_part 	= rec_part
	t.hours 				= h

	return t

def build_lin_test(meta, eq = True, click_percent = 0.2, alpha = 0.1, h = 12, rec_part = 0.2):
	t = TestMetadata(meta)
	t.alpha				= alpha
	t.equalize_clicks 	= eq
	t.click_percent		= click_percent
	t.train_part 			= rec_part
	t.recommendation_part 	= rec_part
	t.hours 				= h

	return t

def get_lin_tests(meta):
	tests = list()
	for alpha in [0.1, 0.01, 0.001]:
		for rec_part in [0.02, 0.05, 0.1, 0.2, 0.5]:
			for click_percent in [0, 0.2, 0.5]:
				eq = click_percent != 0
				tests.append(build_regression_test(meta, eq = eq, click_percent = click_percent, alpha = alpha, h = 12, rec_part = rec_part))
	return tests

def get_nn_tests(meta):
	return [
			# test learning rate
	# build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, eq = True, click_percent = 0.2), 
	# build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.2, eq = True, click_percent = 0.2),
	# build_nn_test(meta, learning_rate = 0.0001, h = 12, rec_part = 0.2, eq = True, click_percent = 0.2),
	
	# test train part
	build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.1, eq = True, click_percent = 0.2),
	build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.2, eq = True, click_percent = 0.2),
	build_nn_test(meta, learning_rate = 0.01,  h = 12, rec_part = 0.5, eq = True, click_percent = 0.2),
	

	# click percent
	build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, eq = True, click_percent = 0.5), 
	build_nn_test(meta, learning_rate = 0.001,  h = 12, rec_part = 0.2, eq = True, click_percent = 0.5),
	build_nn_test(meta, learning_rate = 0.0001, h = 12, rec_part = 0.2, eq = True, click_percent = 0.5),
	build_nn_test(meta, learning_rate = 0.01,   h = 12, rec_part = 0.2, eq = False) 
		]

def get_gp_tests(meta):
	return [	
# test no eq of clicks
build_gp_test(meta, eq = False, nu = 1.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
build_gp_test(meta, eq = False, nu = 2.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
build_gp_test(meta, eq = False, nu = 3.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),

# test eq clicks with 50%
build_gp_test(meta, eq = True, click_percent = 0.5, nu = 1.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
build_gp_test(meta, eq = True, click_percent = 0.5, nu = 2.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
build_gp_test(meta, eq = True, click_percent = 0.5, nu = 3.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),


		# build_gp_test(meta, nu = 1.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
		# build_gp_test(meta, nu = 2.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
		# build_gp_test(meta, nu = 3.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
		# build_gp_test(meta, nu = 1.5, length_scale = 200, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
		# build_gp_test(meta, nu = 2.5, length_scale = 200, cluster_count = 10, alpha = 1, h = 12, rec_part = 0.2),
		# build_gp_test(meta, nu = 1.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 4,  rec_part = 0.2),
		# build_gp_test(meta, nu = 2.5, length_scale = 100, cluster_count = 10, alpha = 1, h = 4,  rec_part = 0.2),
		# build_gp_test(meta, nu = 1.5, length_scale = 100, cluster_count = 20, alpha = 1, h = 12, rec_part = 0.2),
		# build_gp_test(meta, nu = 2.5, length_scale = 100, cluster_count = 20, alpha = 1, h = 12, rec_part = 0.2)
			]