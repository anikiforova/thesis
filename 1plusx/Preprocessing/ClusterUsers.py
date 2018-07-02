import numpy as np 
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from pandas import read_csv

user_dimension = 100
cluster_count = 100
# leaning_size = 100000
# users = np.array([])

def get_e_distance(x, y):
	return np.sqrt(np.dot(x, x) - 2 * np.dot(x, y) + np.dot(y, y))

def get_embedding(user_embedding, clusters):
	user_to_cluster = np.apply_along_axis(lambda c: get_e_distance(user_embedding, c), 1, clusters)
	user_norm = user_to_cluster / np.linalg.norm(user_to_cluster)
	return user_norm

campaign_id = "809153"
users_processed_path = "../../RawData/Campaigns/{0}/Processed/"

users_file_path = users_processed_path.format(campaign_id) + "all_users.csv"
clusters_file_path = users_processed_path.format(campaign_id) + "all_users_clusters.csv"
clustered_users_output_path = users_processed_path.format(campaign_id) + "all_users_clustered.csv"

# read users and clusters
users = read_csv(users_file_path, ",")
clusters = read_csv(clusters_file_path, ",")
clusters['Cluster'] = clusters['Cluster'].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
clusters = [item for sublist in clusters['Cluster'] for item in sublist]
clusters = np.array(clusters).reshape([cluster_count, user_dimension])

# assign users to clusters and give relative distance to clusters
users['UserEmbedding'] = users['UserEmbedding'].apply(lambda x: np.fromstring(x[1:-1], sep=" "))
users["ClusteredEmbedding"] = users["UserEmbedding"].apply(lambda x: get_embedding(x, clusters))

# output users to file using clustered embedding inctead of the user embedding
output = open(clustered_users_output_path, "w")
output.write("UserEmbedding,UserHash\n")
print("Starting output")
for index, row in users.iterrows():
	user_str = np.array2string(row["ClusteredEmbedding"], separator=' ')[1:-1].replace('\n', '')
	output.write("{0},{1}\n".format(user_str, row["UserHash"]))
	output.flush()

output.close()
