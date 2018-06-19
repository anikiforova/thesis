import numpy as np
import random 
import math
import time
from pandas import read_csv
from pandas import DataFrame

from SEFirst import SEFirst
from SEGreedy import SEGreedy 
from SRandom import SRandom 
from SBase import Regressor
from ErrorAdder import * # bad idea but for the sake of simplicity

user_dimension = 100

update_theta = False #True
update_limit = False

poly = 3

file_count = 6
def get_means():
	return np.array(list(map(lambda i: random.uniform(-1, 1), range(0, user_dimension))))

def get_thetas():
	thetas = list()
	for i in range(0, poly):
		theta = get_means()
		theta_norm = np.linalg.norm(theta)
		theta = theta/theta_norm
		thetas.append(theta)

	return thetas

def store_thetas(thetas):
	output = open("../../1plusx/theta.csv", "w")
	for theta in thetas:
		theta_str = np.array2string(theta, separator=',')[1:-1].replace('\n', '')
		output.write(theta_str + "\n")
	output.close()

def store_limit(limit):
	output = open("../../1plusx/limit.csv", "w")
	output.write("{:04}\n".format(limit))
	output.close()

def read_thetas():
	input = open("../../1plusx/theta.csv", "r")
	thetas = list()
	for line in input:
		theta = np.fromstring(line, sep=",")
		thetas.append(theta)
	input.close()
	return thetas	

def read_limit():
	input = open("../../1plusx/limit.csv", "r")
	t = float(input.readline())
	input.close()
	return t	

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def estimate_click(user, thetas):
	return calculate_click_poly2_sigmoid(user, thetas)

# U_t U t1 + U * t2
def calculate_click_poly2(user, thetas):
	t1 = thetas[1].reshape([1, len(thetas[1])])
	u1 = user.reshape([len(user),1])
	u2 = user.reshape([1,len(user)])
	t2 = thetas[1].reshape([len(thetas[1]), 1])

	return (t1.dot(u1)).dot(u2.dot(t2)) + user.dot(thetas[0]) + np.random.normal(0, 0.001, 1)[0]

def calculate_click_poly2_sigmoid(user, thetas):
	return sigmoid(calculate_click_poly2(user, thetas))

def calculate_click_dot(user, thetas):
	return user.dot(thetas[0])

def calculate_click_sigmoid(user, thetas):
	return sigmoid(calculate_click_dot(user, thetas))

def get_data(number_of_files):
	file = "../../1plusx/spart-000{:02}-9492a20a-812b-4f35-92fa-8f8d9aca22e4-c000.csv"  

	users = DataFrame()
	
	for file_index in range(0, number_of_files):
		file = file.format(file_index)
		frame = read_csv(file, sep=',',  header='infer')
		users = users.append(frame)
		update_limit = True

	users = users.values
	print("The number of users in total: {0}".format(len(users)))

	if update_theta:
		thetas = get_thetas()
		store_thetas(thetas)
		print("WARNING: Updated theta.")
	thetas = read_thetas()

	return users, thetas

def get_clicks(users, thetas, percentile, to_randomize_clicks):
	click_estimators = np.array(list(map(lambda user: estimate_click(user, thetas), users)))
	click_estimators_with_error = add_error(click_estimators)
	limit_value = np.percentile(click_estimators_with_error, percentile, axis=0)
	clicks = evaluate_clicks(click_estimators_with_error, limit_value)
	clicks = randomize_clicks(clicks, to_randomize_clicks)
	return clicks

def run_algo(algo, output, thetas, clicks, overall_ctr, repetition):
	click_count = 0.0
	impression = 0.0
	algo_name = type(algo).__name__	
	simulation_impressions = len(clicks) * ctr * 0.7
	print("Starting running {0} with alpha: {1} ctr: {2} rep {3}".format(algo_name, algo.get_alpha(), overall_ctr, repetition))
	print("Running for {0} iterations".format(simulation_impressions))
	while impression < simulation_impressions:
		impression += 1
		selected_user = algo.select()
		algo.update(selected_user, clicks[selected_user])

		click_count += clicks[selected_user]
		# print('.', end='', flush=True)
		if impression % 100 == 0:
			output.write('{:d},{:d},{},{:.2f},{},{},{}\n'.format(int(click_count), int(impression), algo_name, algo.get_alpha(), overall_ctr, repetition, to_random))
			print('Done with: {:.1%} CTR: {:.3%} C: {} I:{}'.format(impression/simulation_impressions, click_count/impression, click_count, impression))
			output.flush()

# evaluation_functions = {}
# alphas = {}
# error_functions = {}
# algorithms = {}
overall_ctr = [0.02,0.005]
repetitions = 10
to_randomize_clicks = [True] # , False
alphas = np.arange(0.0, 0.2, 0.05)
# regressor = Regressor.SGDClassifier
regressor = Regressor.LinearRegression

print("Reading the data...")

users, thetas = get_data(file_count)

output = open("./Results/Poly_Sigmoid_Alpha.csv", "a")
output.write("ClickCount,Impressions,AlgoName,Alpha,OverallCTR,Repetition,Randomized\n")
print("Starting the simulation...")
for ctr in overall_ctr:
	percentile = 100 - ctr * 100
	for to_random in to_randomize_clicks:
		clicks = get_clicks(users, thetas, percentile, to_random)

		# algo_random = SRandom(0.05, users)
		# run_algo(algo_random, output, thetas, clicks, ctr, 0)
		simulation_impressions = len(clicks) * ctr * 0.7
		for rep in range(0, repetitions):
			for alpha in alphas:
				start = time.time()
				algo_egreedy = SEGreedy(alpha, users, regressor, simulation_impressions)
				run_algo(algo_egreedy, output, thetas, clicks, ctr, rep)
				end1 = time.time()
				
				# algo_efirst = SEFirst(0.05, users, regressor, simulation_impressions)
				# run_algo(algo_efirst, output, thetas, clicks, ctr, rep)
				end2 = time.time()

				print("Elapsed for EGreedy: {0} EFirst {1}".format(end1 - start, end2 - end1))


output.close()

