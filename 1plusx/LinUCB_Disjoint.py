import numpy as np
import math
from numpy.linalg import inv

from AlgoBase import AlgoBase
from Metadata import Metadata

class LinUCB_Disjoint(AlgoBase):
	
	def __init__(self, meta):
		super(LinUCB_Disjoint, self).__init__(meta)

	def setup(self):
		super(LinUCB_Disjoint, self).setup()

		self.A 		= np.identity(self.meta.meta.dimensions)
		self.A_i	= np.identity(self.meta.meta.dimensions)
		self.b 		= np.zeros(self.meta.meta.dimensions)
			
	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = self.prepareClicks(users, clicks)
		train_user_count = len(clicks)

		for user_id, click in zip(users, clicks):
			embedding = self.user_embeddings[user_id]
			self.A += embedding.reshape([self.meta.meta.dimensions, 1]).dot(embedding.reshape([1, self.meta.meta.dimensions]))
			if click == 1 :
				self.b += embedding

		self.A_i = inv(self.A)
		theta = self.A_i.dot(self.b) # [self.d, self.d] x [self.d, 1] = [self.d, 1]

		for index, embedding in enumerate(self.user_embeddings):
			mean = embedding.dot(theta)
			var = math.sqrt(embedding.reshape([1, self.meta.meta.dimensions]).dot(self.A_i).dot(embedding))

			self.prediction[index] = mean + self.meta.alpha * var

		self.predictionPosprocessing(users, clicks)		
		print(" Done.")
