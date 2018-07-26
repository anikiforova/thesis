
import numpy as np
from sklearn.cluster import KMeans
import random 
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics.pairwise import euclidean_distances
#from sklearn.preprocessing import normalize

clusters = 10
leaning_size = 100000
user_dimension = 100

campaign_id = "837817" #"809153"
users_processed_path = "../../RawData/Campaigns/{0}/Processed/"

infile_pattern = users_processed_path.format(campaign_id) + "all_users.csv"
outfile_pattern = users_processed_path.format(campaign_id) + "all_users_clusters_{0}.csv".format(clusters)
outfile_pattern_user_assignment = users_processed_path.format(campaign_id) + "all_users_cluster_assignment_{0}.csv".format(clusters)

print("Reading user information.")
users = read_csv(infile_pattern, ",")
users_to_cluster = users.sample(leaning_size, replace=False)
user_embeddings = users_to_cluster['UserEmbedding'].apply(lambda x: np.fromstring(x[1:-1], sep=" ")).values
user_embeddings = [item for sublist in user_embeddings for item in sublist]
user_embeddings = np.array(user_embeddings).reshape([leaning_size, user_dimension])

print("Starting clustering.")
mbk = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
mbk.fit(user_embeddings)

print("Output cluster centers")
output = open(outfile_pattern, "w")
output.write("Cluster\n")
for c in mbk.cluster_centers_:
	c_str = np.array2string(c, precision=7, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
	output.write(c_str + "\n")

output.close()

print("Parsing all user info..")
all_users_embeddings = users['UserEmbedding'].apply(lambda x: np.fromstring(x[1:-1], sep=" ")).values
all_users_embeddings = [item for sublist in all_users_embeddings for item in sublist]
all_users_embeddings = np.array(all_users_embeddings).reshape([users.shape[0], user_dimension])

print("Starting user prediction..")
all_users_cluster_assignment = mbk.fit_predict(all_users_embeddings)

print("Output user assignemnt..")
cluster_counts = np.zeros(clusters)
output_assignment = open(outfile_pattern_user_assignment, "w")
output_assignment.write("ClusterId\n")
for assignemnt in all_users_cluster_assignment:
	output_assignment.write("{0}\n".format(assignemnt))
	cluster_counts[assignemnt] += 1
output_assignment.close()

print("Final assignment:")
print(cluster_counts)