import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase
from Metadata import Metadata

class LinUCB_Disjoint(AlgoBase):
	
	def __init__(self, meta):
		super(LinUCB_Disjoint, self).__init__(meta)

	def setup(self, testMeta):
		super(LinUCB_Disjoint, self).setup(testMeta)

		self.A 		= np.identity(self.meta.dimensions)
		self.A_i	= np.identity(self.meta.dimensions)
		self.b 		= np.zeros(self.meta.dimensions)
		self.theta  = np.zeros(self.meta.dimensions)
			
	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks)
		train_user_count = len(clicks)

		for user_id, click in zip(users, clicks):
			embedding = self.user_embeddings[user_id]
			self.A += embedding.reshape([self.meta.dimensions, 1]).dot(embedding.reshape([1, self.meta.dimensions]))
			if click == 1 :
				self.b += embedding

		self.A_i = inv(self.A)
		self.theta = self.A_i.dot(self.b) # [self.d, self.d] x [self.d, 1] = [self.d, 1]

		self.update_prediction()

		self.predictionPosprocessing(users, clicks)		
		print(" Done.")

	def update_prediction(self):
		print("Updating predictions..")
		for index, embedding in enumerate(self.user_embeddings):
			mean = embedding.dot(self.theta)
			var = math.sqrt(embedding.reshape([1, self.meta.dimensions]).dot(self.A_i).dot(embedding))

			self.prediction[index] = mean + self.testMeta.alpha * var		

		


