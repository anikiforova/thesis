
import pyGPs
import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase

class GP_FITC(AlgoBase):
	
	def __init__(self, alpha, user_embeddings, user_ids, cluster_embeddings, dimensions, filter_clickers = False, soft_click = False):
		print("Starting GP_Clustered setup ...", end='', flush=True)	
		super(GP_Clustered, self).__init__(alpha, user_embeddings, user_ids, filter_clickers, soft_click)

		self.cluster_embeddings = cluster_embeddings
		self.cluster_count = len(self.cluster_embeddings)
		self.d = dimensions
		
		self.o_users = np.array([], dtype=np.uint32)
		self.o_clicks = np.array([], dtype=np.uint32)

		model = pyGPs.GPC_FITC() 

		print("Done.")

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(GP_Clustered, self).prepareClicks(users, clicks)
		
		# self.o_users = np.append(self.o_users, users)
		# self.o_clicks = np.append(self.o_clicks, clicks)
		# cur_users = self.user_embeddings[self.o_users]
		cur_users = self.user_embeddings[users]

		model.setData(cur_users, clicks)
		model.setPrior(inducing_points = self.cluster_embeddings)
		model.optimize()

 		self.prediction = model.predict(self.user_embeddings)

		super(GP_Clustered, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

	def get_recommendations(self, count):
		recommendation_ids = self.predition.argsort()[-count:][::-1]
		recommendation_hashes = [ self.user_id_to_hash[x] for x in recommendation_ids ]

		return set(recommendation_hashes)









