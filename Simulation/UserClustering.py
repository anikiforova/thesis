import numpy as np
from sklearn.cluster import MiniBatchKMeans

user_dimension = 100
clusters = 10
users = np.array([])
user_count = 0

input = open('./Data.csv', "r")
input.readline() # skip header

for line in input:
	line_split = line.split(",")
	user = np.fromstring(line_split[0], dtype=float, sep=' ')
	users = np.append(users, user)
	user_count += 1 

input.close()
print(user_count)
users = users.reshape([user_count, user_dimension])
users = users.transpose()

print("Done with reading data.", flush=True)
print("Starting clustering.", flush=True)

batch_size = 1000
# find the best cluster number
best_score = float('inf')
best_nclusters = 0
for nclusters in range(4, 11):
	mbk = MiniBatchKMeans(init='k-means++', n_clusters=nclusters, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
	mbk.fit(users)
	score = mbk.score(users)
	if score < best_score:
		best_score = score
		best_nclusters = nclusters
	print('Centers = {0}; FPC = {1:.3f}'.format(nclusters, score), flush=True)


# print ("Done learning clusters. Starting updating data.")
# output = open('./ClusteredData.csv', "w")
# input = open('./Data.csv', "r")
# header = input.readline() # skip header
# output.write(header)

mbk = MiniBatchKMeans(init='k-means++', n_clusters=nclusters, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0)
transformed_users = mbk.fit_transform(users)
print(transformed_users[0])
# index = 0
# for line in input:
# 	line_split = line.split(",")
	
# 	user = np.array([])
# 	for i in range(0, best_cluster_size):
# 		user = np.append(user, u[i][index])
	
# 	user_str = np.array2string(user, precision=3, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
# 	output.write('{0},{1},{2}\n'.format(user_str, line_split[1], line_split[2]))
# 	index += 1
	

# input.close()
# output.close()