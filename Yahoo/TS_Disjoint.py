import numpy as np
import math
from numpy.linalg import inv

from Util import to_vector

class TS_Disjoint:
	
	def __init__(self, alpha):
		self.d = 6
		self.alpha = alpha
		self.training_size = 10000
		self.warmup_impressions = 0

		self.A = dict()
		self.A_i = dict()
		self.b = dict()
		self.L = dict()

	def add_new_article(self, article_id):
		if article_id not in self.A:
			self.A[article_id] = np.identity(self.d)
			self.A_i[article_id] = np.identity(self.d)
			self.b[article_id] = np.zeros(self.d)

	def update(self, user, selected_article, click):
		if selected_article in self.A:
			self.A[selected_article] += user.reshape([self.d, 1]).dot(user.reshape([1, self.d]))
			self.A_i[selected_article] = inv(self.A[selected_article])
			if click == 1 :
				self.b[selected_article] += user
	
	def warmup(self, file):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		selected_article = -1
		warmup = False

		if self.warmup_impressions < self.training_size:
			self.warmup_impressions += 1
			self.add_new_article(pre_selected_article)
			self.update(user, pre_selected_article, click)
			selected_article = pre_selected_article
			warmup = True
		else:
			limit = 0.0
			for line in lines:
				article_id = int(line.split(" ")[0])
				self.add_new_article(article_id)
				cur_A_i = self.A_i[article_id]
				cur_b = self.b[article_id]

				cur_theta = cur_A_i.dot(cur_b)
				cur_limit = np.random.normal(user.dot(cur_theta), self.alpha * user.reshape([1, self.d]).dot(cur_A_i).dot(user))
				# cur_limit = user.dot(cur_theta) + self.alpha * math.sqrt(user.reshape([1, self.d]).dot(cur_A_i).dot(user))

				if(cur_limit > limit):
					selected_article = article_id
					limit = cur_limit
		return selected_article, warmup





