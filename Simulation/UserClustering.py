import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
user_dimension = 100
leaning_size = 100000
users = np.array([])

def get_e_distance(x, y):
	return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

input = open('./Data.csv', "r")
input.readline() # skip header

user_count = 0
for line in input:
	line_split = line.split(",")
	user = np.fromstring(line_split[0], dtype=float, sep=' ')
	users = np.append(users, user)
	user_count += 1
	if user_count >= leaning_size:
		break
input.close()

users = users.reshape([user_dimension, user_count])
users = users.transpose()

print("Done with reading data.", flush=True)
print("Starting clustering.", flush=True)

# find the best cluster number
best_score = float('inf')
best_nclusters = 5

for nclusters in range(4, 11):
	mbk = KMeans(init='k-means++', n_clusters=nclusters, n_init=10)
	mbk.fit(users)
	score = - mbk.score(users)
	if score < best_score:
		best_score = score
		best_nclusters = nclusters
	print('Centers = {0}; FPC = {1:.3f}'.format(nclusters, score), flush=True)


print ("Done learning clusters. Starting updating data.")

output = open('./ClusteredData.csv', "w")
input = open('./Data.csv', "r")
header = input.readline() # skip header
output.write(header)

mbk = KMeans(init='k-means++', n_clusters=best_nclusters, n_init=10)
mbk.fit(users)

cluster_centers = mbk.cluster_centers_

index = 0
for line in input:
	line_split = line.split(",")
	user = np.fromstring(line_split[0], sep=" ")
	user_to_cluster = np.apply_along_axis(lambda c: get_e_distance(user, c), 1,cluster_centers)
	user_norm = user_to_cluster / np.linalg.norm(user_to_cluster)
	user_str = np.array2string(user_norm, precision=3, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
	output.write('{0},{1},{2}\n'.format(user_str, line_split[1], line_split[2]))
	index += 1
	if index % 10000 == 0:
		print('.', end='', flush=True)

input.close()
output.close()