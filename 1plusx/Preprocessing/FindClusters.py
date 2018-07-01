
import numpy as np
from sklearn.cluster import KMeans
import random 
from pandas import read_csv
from pandas import DataFrame

clusters = 100
leaning_size = 1000
user_dimension = 100

campaign_id = "809153"
users_processed_path = "../../RawData/Campaigns/{0}/Processed/"

infile_pattern = users_processed_path.format(campaign_id) + "all_users.csv"
outfile_pattern = users_processed_path.format(campaign_id) + "all_users_clusters.csv"

print("Starting clustering.")

users = np.array([])

users = read_csv(infile_pattern, ",")
users_to_cluster = users.sample(leaning_size, replace=False)
users_to_cluster["UserEmbedding"] = users_to_cluster['UserEmbedding'].apply(lambda x: np.fromstring(x[1:-1], sep=","))

user_embeddings = np.array(users_to_cluster["UserEmbedding"])
print(user_embeddings)
#.reshape([leaning_size, user_dimension])
print(user_embeddings.shape)

# users = users.reshape([user_count, user_dimension])
mbk = KMeans(init='k-means++', n_clusters=clusters, n_init=10)
mbk.fit(users)
print("Done with clustering")

# output = open("../../1plusx/Clusters.csv", "w")
for c in mbk.cluster_centers_:
	print(c)
# 	c_str = np.array2string(c, precision=7, separator=' ', suppress_small=True)[1:-1].replace('\n', '')
# 	output.write(c_str + "\n")

# output.close()




