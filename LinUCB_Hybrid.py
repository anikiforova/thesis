import numpy as np
import math
import re
from numpy.linalg import inv

from util import to_vector

class LinUCB_Hybrid:
	
	def __init__(self):
		self.d = 6
		self.k = 36
		self.alpha = 0.4

		self.A0 = np.identity(self.k)
		self.b0 = np.zeros(self.k)
		self.A0_i = np.identity(self.k)
		self.beta = self.A0_i.dot(self.b0)

		self.A = dict()
		self.A_i = dict()
		self.B = dict()
		self.b = dict()
		self.articles = dict()
		self.bad_articles = set()

	def update(self, user, selected_article, click):
		article = self.articles[selected_article]
		
		z = np.outer(user, article)
		z_1_k = z.reshape([1, self.k])
		z_k_1 = z.reshape([self.k, 1])
		user_1_d = user.reshape([1, self.d])
		user_d_1 = user.reshape([self.d, 1])
		
		self.A[selected_article] += user_d_1.dot(user_1_d) 
		self.B[selected_article] += user_d_1.dot(z_1_k)
		self.A0 += z_k_1.dot(z_1_k)

		self.A_i[selected_article] = inv(self.A[selected_article])
		self.A0_i = inv(self.A0)

		if click == 1:
			self.b[selected_article] += user
			self.b0 += z.reshape([self.k])

	def select(self, user, lines, exploit):
		limit = 0.0
		selected_article = 0
		for line in lines:
			article_id = int(line.split(" ")[0])
			
			if article_id in self.bad_articles:
				continue;	

			article = np.zeros([1, self.d])
			if article_id not in self.A:
				try:
					article = to_vector(line)
				except IndexError:
					print("Skipping line, weird formatting.." + str(article_id))
					self.bad_articles.add(article_id)
					continue; 

				self.articles[article_id] = article
				self.A[article_id] = np.identity(self.d)
				self.A_i[article_id] = np.identity(self.d)
				self.B[article_id] = np.zeros(self.d * self.k).reshape([self.d, self.k])
				self.b[article_id] = np.zeros(self.d)

			z = np.outer(user, article).reshape([1, self.k])
			cur_A_i = self.A_i[article_id]
			cur_B = self.B[article_id]
			cur_b = self.b[article_id]
		
			cur_theta = cur_A_i.dot((cur_b - cur_B.dot(self.beta))) 

			if exploit:
				cur_limit = z.dot(self.beta) + user.dot(cur_theta)
			else:
				pre_user_A_i = user.dot(cur_A_i) 
				pre_zT_A0_i = z.dot(self.A0_i)
				pre_A_i_user = cur_A_i.dot(user)
			
				cur_s = pre_zT_A0_i.dot(z.reshape([self.k, 1])) - 2 * pre_zT_A0_i.dot(cur_B.T).dot(pre_A_i_user) + pre_user_A_i.dot(user) + pre_user_A_i.dot(cur_B).dot(self.A0_i).dot(cur_B.T).dot(pre_A_i_user)

				cur_limit = z.dot(self.beta) + user.dot(cur_theta) + self.alpha * math.sqrt(cur_s)

			if(cur_limit >= limit):
				selected_article = article_id
				limit = cur_limit

		# print(selected_article)
		return selected_article





