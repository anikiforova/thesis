import numpy as np
import math
import random
from numpy.linalg import inv

class EGreedy_Disjoint:
	
	def __init__(self, alpha, cluster_count):
		self.d = 100
		self.alpha = alpha
		self.cluster_count = cluster_count
		
		self.A = self.get_clusters()
		self.A_i = np.zeros(self.cluster_count)
		self.b = np.zeros(self.cluster_count)

	def get_clusters(self):
		clusters_input = open("../../1plusx/Clusters.csv", "r")
		cluster_centers = np.array([])
		cluster_count = 0

		for line in clusters_input:
			center = np.fromstring(line[1:-1], sep=" ")
			cluster_centers = np.append(cluster_centers, center)
			cluster_count += 1
		cluster_centers = cluster_centers.reshape([cluster_count, self.d])
		clusters_input.close()

		return cluster_centers

	def update(self, user, selected_article, click):
		self.A[selected_article] += user.reshape([self.d, 1]).dot(user.reshape([1, self.d]))
		self.A_i[selected_article] = inv(self.A[selected_article])
		if click == 1 :
			self.b[selected_article] += user
	
	def warmup(self, file):
		pass

	def select(self, user, pre_selected, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha
		
		selected_article = -1
		
		if explore:
			selected_article = randint(0, self.clusters-1)
		else:
			limit = 0.0
			for c in self.clusters:
				cur_limit = user.dot(self.A_i[c].dot(self.b[c]))
				if cur_limit > limit:
					selected_article = article_id
					limit = cur_limit

		return selected_article, False





