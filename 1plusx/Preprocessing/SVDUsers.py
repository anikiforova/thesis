import numpy as np

from pandas import read_csv
from pandas import DataFrame

leaning_size = 100000
user_dimension = 100
new_user_dimension = 100
total_user_count = 2916809

campaign_id = "809153"
# users_processed_path = "../RawData/Campaigns/{0}/Processed/"
users_processed_path = "../../RawData/Campaigns/{0}/Processed/"

infile_pattern = users_processed_path.format(campaign_id) + "all_users.csv"
outfile_pattern = users_processed_path.format(campaign_id) + "all_users_svd_{0}.csv".format(new_user_dimension)

print("Reading user information.")
users_to_cluster = read_csv(infile_pattern, ",")
# users_to_cluster = read_csv(infile_pattern, ",", nrows=leaning_size)
# users_to_cluster = users_to_cluster.sample(leaning_size, replace=False)

user_embeddings = users_to_cluster['UserEmbedding'].apply(lambda x: np.fromstring(x[1:-1], sep=" ")).values
user_embeddings = [item for sublist in user_embeddings for item in sublist]
# user_embeddings = np.array(user_embeddings).reshape([leaning_size, user_dimension])
user_embeddings = np.array(user_embeddings).reshape([total_user_count, user_dimension])

print("Starting SVD.")
u, s, vh = np.linalg.svd(user_embeddings, full_matrices=False)
# use only the first 10 eigenvalues and remove other information

# reduce the dimentionality to 10
s1 = s[0:new_user_dimension] 
u1 = u[0:total_user_count, 0:new_user_dimension]
reconstructed_users = (u1 * s1)#.dot(vh)
# print(reconstructed_users.shape)
print("Output new user cluster info..")
# output users to file using clustered embedding inctead of the user embedding
output = open(outfile_pattern, "w")
output.write("UserEmbedding,UserHash\n")
index = 0
user_hashes = np.array(users_to_cluster["UserHash"])
for user_hash, row in zip(user_hashes, reconstructed_users):
	user_str = np.array2string(row, separator=' ')[1:-1].replace('\n', '')
	output.write("{0},{1}\n".format(user_str, user_hash))
	output.flush()
	index += 1
	if index % 100000 == 0: 
		print(index) 

output.close()







