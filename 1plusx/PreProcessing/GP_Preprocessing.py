
import numpy as np
import random 
import math
import time
import datetime
from pandas import read_csv
from pandas import DataFrame
from numpy import genfromtxt


def normalize_dimension(dimension):
	min_value = np.min(dimension)
	max_value = np.max(dimension)
	return (dimension - min_value)/(max_value - min_value)

def squared_distance(x, y):
	return sqrt(np.sum((x-y)*(x-y)))

def kernel(self, x, y, d):
	distance = squared_distance(x, y)
	return 0.5 * np.exp(- (distance * distance)/(2*(d**2)))

def kernel_from_distance(self, distance):
	return np.exp(-0.5 * distance)

def calculate_kernel(K, cluster_count, cluster_embeddings):
	for x in np.arange(0, cluster_count):
		for y in np.arange(0, cluster_count):
			K[x][y] = kernel(cluster_embeddings[x], cluster_embeddings[y]) 

def calculate_users_to_clusters(cluster_count, user_count, user_embeddings, cluster_embeddings):
	cluster_counts = np.zeros(cluster_count)

	for user_index in np.arange(0, user_count):
		distances = [squared_distance(user_embeddings[user_index], c, cluster_count) for c in cluster_embeddings]
		# self.K_u_c[user_index] = np.array([self.kernel_from_distance(d) for d in distances])
		# self.u_to_c[user_index] = int(np.argmax(distances)) 
		closest_cluster_id = int(np.argmax(distances))
		cluster_counts[closest_cluster_id] += 1
		# self.k_u_u[user_index] = self.kernel(self.user_embeddings[user_index], self.user_embeddings[user_index]) 

	print("Clusters assignment: ")
	print(cluster_counts)

dimensions 		= 100
cluster_count 	= 10
noiseVar 		= 0.1

clusters =""# "_svd_" + str(dimensions)
algoName = "GP_Clustered"

name = "809153"
path = "..//..//RawData//Campaigns"

print("Reading users.. ", end='', flush=True)	
users = read_csv("{0}//{1}//Processed//all_users{2}.csv".format(path, name, clusters), header=0)#, index_col=1)
print(" Done.")

print("Parsing users.. ", end='', flush=True)	
user_ids = np.array(users["UserHash"])
user_embeddings = np.array(users["UserEmbedding"])
user_embeddings = np.array([np.fromstring(x, sep=" ") for x in user_embeddings]).reshape([len(user_ids), dimensions])
print(" Done.")

print("Normalizing users .. ", end='', flush=True)	
user_embeddings = np.apply_along_axis(lambda dimension: normalize_dimension(dimension), 0, user_embeddings)
print(" Done.")

clusters = read_csv("{0}//{1}//Processed//all_users_clusters_{2}.csv".format(path, name, cluster_count), header=0)
cluster_embeddings = np.array(clusters["Cluster"])
cluster_embeddings = np.array([np.fromstring(x, sep=" ") for x in cluster_embeddings]).reshape([cluster_count, dimensions])
cluster_embeddings 	= np.apply_along_axis(lambda dimension: normalize_dimension(dimension), 0, cluster_embeddings)

user_count = len(user_ids)

# K = np.identity(cluster_count)
# calculate_kernel(K, cluster_count, cluster_embeddings)
# K += np.identity(cluster_count) * noiseVar

K_u_c = np.zeros(user_count * cluster_count).reshape([user_count, cluster_count])
u_to_c = np.zeros(user_count, dtype=int)
k_u_u = np.zeros(user_count)
calculate_users_to_clusters(cluster_count, user_count, user_embeddings, cluster_embeddings)

# K_output 		= open("{0}//{1}//Processed//GP//K_{2}.csv".format(path, name, cluster_count), "w")
# K_u_c_output 	= open("{0}//{1}//Processed//GP//K_u_c_{2}.csv".format(path, name, cluster_count), "w")
# u_to_c_output	= open("{0}//{1}//Processed//GP//u_to_c_{2}.csv".format(path, name, cluster_count), "w")
# k_u_u_output 	= open("{0}//{1}//Processed//GP//k_u_u_{2}.csv".format(path, name, cluster_count), "w")








