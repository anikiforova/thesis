import numpy as np
import skfuzzy as fuzz
# from sklearn.cluster import KMeans

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

find the best cluster number
for ncenters in range(5, 10):
	cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(users, ncenters, 2, error=0.005, maxiter=1000, init=None)
	print('Centers = {0}; FPC = {1:.3f}'.format(ncenters, fpc), flush=True)

best_cluster_size = 5
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(users, best_cluster_size, 2, error=0.005, maxiter=1000, init=None)


print ("Done learning clusters. Starting updating data.")
output = open('./ClusteredData.csv', "w")
input = open('./Data.csv', "r")
header = input.readline() # skip header
output.write(header)

u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(users, cntr, 2, error=0.005, maxiter=1000)
print(u.shape)

index = 0
for line in input:
	line_split = line.split(",")
	
	user = np.array([])
	for i in range(0, best_cluster_size):
		user = np.append(user, u[i][index])
	
	user_str = np.array2string(user, precision=3, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
	output.write('{0},{1},{2}\n'.format(user_str, line_split[1], line_split[2]))
	index += 1
	

input.close()
output.close()