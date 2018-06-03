
import numpy as np
from sklearn.cluster import KMeans
import random 

size = 100
percentile = 95
simulation_impressions = 10000.0
total_impressions = 1000000.0
leaning_size = 100000
user_dimension = 100

def get_e_distance(x, y):
	return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def get_variances():
	return np.array(list(map(lambda i: random.uniform(0, 1), range(0, size))))

def get_means():
	return np.array(list(map(lambda i: random.uniform(-1, 1), range(0, size))))

def get_random_within_range(var, mean):
	num = -2
	while -1 > num or num > 1:
		num = np.random.normal(mean, var)
		
	return num

def get_embedding(vars, means):
	return np.array(list(map(lambda i: get_random_within_range(vars[i], means[i]), range(0, size))))
	
def get_theta():	
	theta = get_means()
	theta_norm = np.linalg.norm(theta)
	theta = theta/theta_norm
	theta_str = np.array2string(theta, precision=3, separator=' ', suppress_small=True)[1:-1].replace('\n', '')

	return theta, theta_str

def calculate_click(user, theta):
	user_norm = np.linalg.norm(user)
	normalized_user = user / user_norm
	value = user.dot(theta)
	return value

def get_percentile_through_simulated_data(theta, user_feature_variances, user_feature_means):
	clicks = list()
	for i in range(0, int(simulation_impressions)):
		user = get_embedding(user_feature_variances, user_feature_means)
		click = calculate_click(user, theta)
		clicks.append(click)	
	limit_value = np.percentile(clicks, percentile, axis=0)
	print("Limit value: {0}".format(limit_value))
	return limit_value

user_feature_variances = get_variances()
user_feature_means = get_means()

theta, theta_str = get_theta()
 
infile_pattern = "../../1plusx/part-000{:02}-9492a20a-812b-4f35-92fa-8f8d9aca22e4-c000.csv"  
outfile_pattern = "../../1plusx/clicks_part-000{:02}-9492a20a-812b-4f35-92fa-8f8d9aca22e4-c000.csv"  

print("Starting clustering.")

users = np.array([])
user_count = 0
input = open(infile_pattern.format(0), "r")
for  line in input:
	user = np.fromstring(line[1:-1], sep=",")
	users = np.append(users, user)
	user_count += 1
	if user_count > leaning_size:
		break	
input.close()
users = users.reshape([user_count, user_dimension])
mbk = KMeans(init='k-means++', n_clusters=100, n_init=10)
mbk.fit(users)
cluster_centers = mbk.cluster_centers_
print("Done with clustering")

impressions = 0.0
click_count = 0.0
for i in range(0, 12):

	input = open(infile_pattern.format(0), "r")
	clicks = list()

	for  line in input:
		user = np.fromstring(line[1:-1], sep=",")
		clicks.append(calculate_click(user, theta))	

	limit_value = np.percentile(clicks, percentile, axis=0)
	clicks = list()
	print("Estimation complete. Limit Value: {0}.".format(limit_value))
	input.close()

	input = open(infile_pattern.format(0), "r")
	output = open(outfile_pattern.format(0), "w")
	output.write('User,Ad,Click\n')
	click_count = 0.0

	print("Start file {0}".format(i))
	for line in input:
		impressions += 1
		user = np.fromstring(line[1:-1], sep=",")
		click = calculate_click(user, theta)
		if click >= limit_value:
			click = 1
			click_count+=1
		else:
			click = 0
		
		# print (user.shape)
		# print (cluster_centers.shape)
		user_to_cluster = np.apply_along_axis(lambda c: get_e_distance(user, c), 1,cluster_centers)
		user_norm = user_to_cluster / np.linalg.norm(user_to_cluster)
		user_str = np.array2string(user_norm, precision=7, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
		output.write('{0},{1},{2}\n'.format(user_str, 1, click))
		if impressions % 1000 == 0:
			output.flush()
			print("Impressions done: {0}, Clicks %: {1}".format(impressions/total_impressions, click_count/impressions, flush=True))

	output.close()
	input.close()			
	print("Done file {0}.".format(i))

print("Final percentage: {0}".format(click_count/impressions))



