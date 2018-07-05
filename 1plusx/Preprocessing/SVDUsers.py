import numpy as np
import numpy.linalg.svd as svd

from pandas import read_csv
from pandas import DataFrame

leaning_size = 1000
user_dimension = 100

campaign_id = "809153"
users_processed_path = "../../RawData/Campaigns/{0}/Processed/"

infile_pattern = users_processed_path.format(campaign_id) + "all_users.csv"
outfile_pattern = users_processed_path.format(campaign_id) + "all_users_clusters_10.csv"

print("Reading user information.")
# users = read_csv(infile_pattern, ",")
# users_to_cluster = users.sample(leaning_size, replace=False)
users_to_cluster = read_csv(infile_pattern, ",", nrows=leaning_size)
user_embeddings = users_to_cluster['UserEmbedding'].apply(lambda x: np.fromstring(x[1:-1], sep=" ")).values
user_embeddings = [item for sublist in user_embeddings for item in sublist]
user_embeddings = np.array(user_embeddings).reshape([leaning_size, user_dimension])

print("Starting SVD.")





