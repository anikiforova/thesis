
import numpy as np
import random 

size = 100
percentile = 95
simulation_impressions = 10000.0
impressions = 10000.0

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
	# variances = get_variances()
	# means = get_variances()
	
	# theta = get_embedding(variances, means) 
	theta = get_means()
	theta_norm = np.linalg.norm(theta)
	theta = theta/theta_norm
	theta_str = np.array2string(theta, precision=3, separator=' ', suppress_small=True)[1:-1].replace('\n', '')

	return theta, theta_str

def calculate_click(user, theta):
	user_norm = np.linalg.norm(user)
	normalized_user = user / user_norm
	value = user.dot(theta)
	# assert (-1 >= value and value <= 1, value)
	# print(value)
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
 
limit_value = get_percentile_through_simulated_data(theta, user_feature_variances, user_feature_means)


output = open('./Data.csv', "w")
output.write('User,Ad,Click\n')
click_count = 0.0

for i in range(1, int(impressions)+1):
	user = get_embedding(user_feature_variances, user_feature_means)
	click = calculate_click(user, theta)
	if click >= limit_value:
		click = 1
		click_count+=1
	else:
		click = 0

	user_str = np.array2string(user, precision=3, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
	output.write('{0},{1},{2}\n'.format(user_str, 1, click))
	if i % 1000 == 0:
		output.flush()
		print("Impressions done: {0}, Clicks %: {1}".format(i/impressions, click_count/i, flush=True))

print("Final percentage: {0}".format(click_count/impressions))
output.close()


