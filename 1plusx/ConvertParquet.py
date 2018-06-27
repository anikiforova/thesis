import pandas as pd
import numpy as np

name = "772567-00005"
path = "..//..//RawData//Campaigns"
result = pd.read_parquet("{0}//{1}.parquet".format(path, name), engine='pyarrow')
print("Number of impressions:" + str(result.shape))
	
result = result.rename(index=str, columns={"v":"UserEmbedding", "hash":"UserHash", "kind":"Click"})

result["Click"] = result['Click'].apply(lambda x: ((x == "Click") * 1) + (x == "Conversion") * 2)

result.to_csv("{0}//{1}_impressions.csv".format(path, name), sep=",", header=True, \
	index=False, columns=["UserHash", "Click"])

users = result.groupby(['UserHash'])['UserEmbedding', 'UserHash'].head(1)
print(users.shape)

np.set_printoptions(precision=4)
users['UserEmbedding'] = users['UserEmbedding'].apply(lambda x: np.array(x))

users = users.values
output = open("{0}//{1}_users.csv".format(path, name), "w")
output.write("UserEmbedding,UserHash\n")
for u in users:
	user_str = np.array2string(u[0], separator=' ')[1:-1].replace('\n', '')
	output.write("{0},{1}\n".format(user_str, u[1]))
	output.flush()

output.close()

# import numpy as np
# import pandas as pd
# import pyarrow as pa
# import pyarrow.parquet as pq

# # name = "00030"
# path = "..//..//RawData//Campaigns"
# full_path = "{0}//{1}.parquet".format(path, name)
# parquet_file = pq.ParquetFile(full_path)
