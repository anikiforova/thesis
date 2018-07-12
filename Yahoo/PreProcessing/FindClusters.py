
import numpy as np
from sklearn.cluster import KMeans
import random 
from pandas import read_csv
from pandas import DataFrame

from Util import to_vector

clusters = 10
learning_size = 100000
user_dimension = 6

campaign_id = "809153"
users_processed_path = "../../RawData/Campaigns/{0}/Processed/"

input = open("..//..//..//YahooData//ydata-fp-td-clicks-v1_0.20090501", "r")
output = open("..//..//..//YahooData//user_clusters{0}.csv".format(clusters), "w")

print("Reading user information.")
users = list()
user_count = 0
for line in input:
	user_count += 1
	line = line.split("|")
	user = to_vector(line[1])
	users.append(user)
	if user_count >= learning_size:
		break;

user_embeddings = np.array(users).reshape([user_count, user_dimension])

print("Starting clustering.")
mbk = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
mbk.fit(user_embeddings)

print("Output cluster centers")
output.write("Cluster\n")
for c in mbk.cluster_centers_:
	c_str = np.array2string(c, precision=7, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
	output.write(c_str + "\n")

output.close()




