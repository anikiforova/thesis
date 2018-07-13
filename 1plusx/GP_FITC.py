
# import pyGPs
import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase

class GP_FITC(AlgoBase):
	
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, filter_clickers = False, soft_click = False):
		print("Starting GP_Clustered setup ...", end='', flush=True)	
		super(GP_FITC, self).__init__(user_embeddings, user_ids, filter_clickers, soft_click)

		self.cluster_embeddings = cluster_embeddings
		self.cluster_count = len(self.cluster_embeddings)
		self.d = dimensions
		
		self.o_users = np.array([], dtype=np.uint32)
		self.o_clicks = np.array([], dtype=np.uint32)

		# self.model = pyGPs.GPC_FITC() 

		print("Done.")

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(GP_FITC, self).prepareClicks(users, clicks)
		
		# self.o_users = np.append(self.o_users, users)
		# self.o_clicks = np.append(self.o_clicks, clicks)
		# cur_users = self.user_embeddings[self.o_users]
		cur_users = self.user_embeddings[users]
		print(cur_users.shape)
		print(clicks.shape)
		
		# self.model.setData(cur_users.reshape([len(clicks), self.d]), clicks.reshape([len(clicks), 1]))
		# self.model.setPrior(inducing_points = self.cluster_embeddings)
		# self.model.optimize()

		# self.prediction = self.model.predict(self.user_embeddings)

		super(GP_FITC, self).predictionPosprocessing(users, clicks)		
		print(" Done.")










