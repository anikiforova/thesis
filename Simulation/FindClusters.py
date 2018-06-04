
import numpy as np
from sklearn.cluster import KMeans
import random 

clusters = 100
leaning_size = 100000
user_dimension = 100

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
mbk = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
mbk.fit(users)
print("Done with clustering")

output = open("../../1plusx/Clusters.csv", "w")
for c in mbk.cluster_centers_:
	# print(c)
	c_str = np.array2string(c, precision=7, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
	output.write(c_str + "\n")

output.close()




