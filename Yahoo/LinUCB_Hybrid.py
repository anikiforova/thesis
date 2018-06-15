import numpy as np
import math
import re
from numpy.linalg import inv

from Util import to_vector

class LinUCB_Hybrid:
	
	def __init__(self, alpha):
		self.d = 6
		self.k = 36
		self.alpha = alpha
		self.training_size = 10000
		self.warmup_impressions = 0

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

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if article_id in self.bad_articles:
			return -1

		if article_id not in self.A:
			try:
				article = to_vector(line)
			except IndexError:
				print("Skipping line, weird formatting.." + str(article_id))
				self.bad_articles.add(article_id)
				return -1

			self.articles[article_id] = article
			self.A[article_id] = np.identity(self.d)
			self.A_i[article_id] = np.identity(self.d)
			self.B[article_id] = np.zeros(self.d * self.k).reshape([self.d, self.k])
			self.b[article_id] = np.zeros(self.d)

		return article_id

	def warmup(self, file):
		self.warmup_impressions = 0
		for line in file:
			if self.warmup_impressions > self.training_size:
				break

			self.warmup_impressions += 1
			line = line.split("|")
			no_space_line = line[0].split(" ")
			pre_selected_article = int(no_space_line[1])
			click = int(no_space_line[2])
			user = to_vector(line[1])

			for article_line in line[2:]:
				self.add_new_article(article_line)

			# If preselected article is the one with bad format skip
			if pre_selected_article in self.bad_articles:
				continue;

			self.update(user, pre_selected_article, click)

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

		self.beta = self.A0_i.dot(self.b0)

	def select(self, user, pre_selected_article, lines, exploit, click):
		
		selected_article = -1
		warmup = False # skipping implementation of non warmup version

		limit = 0.0
		for line in lines:
			article_id = self.add_new_article(line)
			if article_id == -1: # invalid article
				continue

			article = self.articles[article_id]
			cur_A_i = self.A_i[article_id]
			cur_B = self.B[article_id]
			cur_b = self.b[article_id]
			z = np.outer(user, article).reshape([1, self.k])

			cur_theta = cur_A_i.dot((cur_b - cur_B.dot(self.beta))) 	
			pre_user_A_i = user.dot(cur_A_i) 
			pre_zT_A0_i = z.dot(self.A0_i)
			pre_A_i_user = cur_A_i.dot(user)
		
			cur_s = pre_zT_A0_i.dot(z.reshape([self.k, 1])) - 2 * pre_zT_A0_i.dot(cur_B.T).dot(pre_A_i_user) + pre_user_A_i.dot(user) + pre_user_A_i.dot(cur_B).dot(self.A0_i).dot(cur_B.T).dot(pre_A_i_user)

			cur_limit = z.dot(self.beta) + user.dot(cur_theta) + self.alpha * math.sqrt(cur_s)

			if(cur_limit >= limit):
				selected_article = article_id
				limit = cur_limit

		# print(selected_article)
		return selected_article, warmup





