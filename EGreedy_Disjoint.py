import numpy as np
import math
from numpy.linalg import inv

from Util import to_vector

class EGreedy_Disjoint:
	
	def __init__(self, alpha):
		self.d = 6
		self.alpha = alpha
		
		self.A = dict()
		self.A_i = dict()
		self.b = dict()

	def add_new_article(self, article_id):
		if article_id not in self.A:
			self.A[article_id] = np.identity(self.d)
			self.A_i[article_id] = np.identity(self.d)
			self.b[article_id] = np.zeros(self.d)

	def update(self, user, selected_article, click):
		self.A[selected_article] += user.reshape([self.d, 1]).dot(user.reshape([1, self.d]))
		self.A_i[selected_article] = inv(self.A[selected_article])
		if click == 1 :
			self.b[selected_article] += user
	
	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha
		
		selected_article = -1
		articles = list()
		
		if explore:
			for line in lines:
				article_id = int(line.split(" ")[0])
				self.add_new_article(article_id)
			selected_article = np.random.choice(articles, 1)

		else:
			limit = 0.0
			for line in lines:
				article_id = int(line.split(" ")[0])
				self.add_new_article(article_id)
				
				cur_limit = user.dot(self.A_i[article_id].dot(self.b[article_id]))
				if cur_limit > limit:
					selected_article = article_id
					limit = cur_limit

		return selected_article, False





