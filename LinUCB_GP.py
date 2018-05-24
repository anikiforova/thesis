import numpy as np
import math
from numpy.linalg import inv
from numpy.linalg import det
from util import to_vector

class LinUCB_GP:
	
	def __init__(self):
		self.d = 6
		self.fixed_beta = 0.05
		self.dimention = 20
		self.bundle_size = 100

		self.bad_articles = set()
		self.users = np.array([])
		self.articles = np.array([])
		self.clicks = np.array([])

		self.K = np.identity(self.dimention)
		self.K_i = np.identity(self.dimention)
		self.all_articles = dict() 

		self.article_relatedness = dict()

	def successfully_add_article_from_line(self, line, article_id):
		try:
			article = to_vector(line)
			self.article_relatedness[article_id] = dict()
			self.article_relatedness[article_id][article_id] = 1
			self.all_articles[article_id] = article

			for a in self.all_articles.keys():
				value = self.exponential_kernel(article, self.all_articles[a])
				self.article_relatedness[article_id][a] = value
				self.article_relatedness[a][article_id] = value

		except IndexError:
			print("Skipping line, weird formatting.." + str(article_id))
			self.bad_articles.add(article_id)
			return False
		return True


	def initialize(self, file):
		selected_articles = list()
		selected_users = list()
		selected_clicks = list()

		for i in range (0, 1000):
			line = file.readline().split("|")
			no_space_line = line[0].split(" ")
			pre_selected_article = int(no_space_line[1])
			click = int(no_space_line[2])
			user = to_vector(line[1])
			
			for line in line[2:]:
				article_id = int(line.split(" ")[0])
			
				if article_id in self.bad_articles:
					continue;	

				if article_id not in self.all_articles:
					self.successfully_add_article_from_line(line, article_id)
			
			selected_articles.append(pre_selected_article)
			selected_users.append(user)
			selected_clicks.append(click)


		self.update_random(selected_users, selected_articles, selected_clicks)


	def exponential_kernel(self, a, b):
		return np.exp(-np.sum((a-b)*(a-b))/(self.d**2))

	def kernel_users(self, user1, user2):
		return self.exponential_kernel(user1, user2)
		
	def kernel_articles(self, article_id1, article_id2):
		return self.article_relatedness[article_id1][article_id2] 		

	def kernel(self, user1, article_id1, user2, article_id2):
		return self.kernel_users(user1, user2) * self.kernel_articles(article_id1, article_id2)

	def get_user_kernel_array(self, user):
		return np.array(list(map(lambda o: self.kernel_users(user, o), self.users)))

	def gp_inference(self, user, user_kernels, article_id):
		article_kernels =  np.array(list(map(lambda o: self.kernel_articles(article_id, o), self.articles)))

		cur_k = user_kernels * article_kernels
		pre_cur_k_K_i = cur_k.dot(self.K_i)
		mu = pre_cur_k_K_i.dot(self.clicks) 
		var = self.kernel(user, article_id, user, article_id) - pre_cur_k_K_i.dot(cur_k.reshape(self.dimention, 1))

		return mu, var

	def beta_inference(self, t, article_id):
		return self.fixed_beta

	def get_bundle_size(self):
		return self.bundle_size

	def get_dimention_size(self):
		return self.dimention

	def select(self, user, lines, exploit, t):
		limit = 0.0
		selected_article = 0
		ucb = 0

		user_kernels = self.get_user_kernel_array(user)
		cur_articles = list()
		for line in lines:
			article_id = int(line.split(" ")[0])
			
			if article_id in self.bad_articles:
				continue;	

			if article_id not in self.all_articles and not self.successfully_add_article_from_line(line, article_id):
				continue

			cur_articles.append(article_id)	
			cur_mu, cur_var = self.gp_inference(user, user_kernels, article_id)
			beta = self.beta_inference(t, article_id)
			if ucb <= cur_mu + beta * cur_var:
				ucb = cur_mu + beta * cur_var
				selected_article = article_id

		if selected_article == 0:
			selected_article = np.random.choice(cur_articles, 1)

		return selected_article

	def select_random(self, user, lines, exploit, t):
		cur_articles = list()
		for line in lines:
			article_id = int(line.split(" ")[0])
			cur_articles.append(article_id)
			
		selected_article = np.random.choice(cur_articles, 1)
		return selected_article

	def update_random(self, selected_users, selected_articles, selected_clicks):
		new_len = len(selected_users) + len(self.users)
		self.users = np.append(self.users, selected_users).reshape(new_len, self.d)
		self.articles = np.append(self.articles, selected_articles)
		self.clicks = np.append(self.clicks, selected_clicks )

		subset_indexes = np.random.choice(range(0, len(self.users)), self.dimention, replace=False)
		self.users 		= self.users[subset_indexes]
		self.articles 	= self.articles[subset_indexes]
		self.clicks 	= self.clicks[subset_indexes]

		for i in range(0, self.dimention):
			for j in range(0, self.dimention):
				self.K[i][j] = self.kernel(self.users[i], self.articles[i], self.users[j], self.articles[j] )
		try :
			self.K_i = inv(self.K)
		except np.linalg.linalg.LinAlgError:
			# print(self.K)
			print("Error singular value")

	def calculate_stripped_entropy(self, users, articles):
		# print(articles)
		dim = len(articles)
		var = np.zeros(dim * dim).reshape([dim, dim])

		for i in range(0, self.dimention):
			for j in range(i, self.dimention):
				# if users[i][0] == users[j][0] and users[i][1] == users[j][1] and users[i][2] == users[j][2] and users[i][3] == users[j][3] and users[i][4] == users[j][4] and articles[i] == articles[j]:
					# print("same " + str(i ) + " " + str(j))
				var[i][j] = self.kernel(users[i], articles[i], users[j], articles[j])
				var[j][i] = var[i][j]

		# print(det(var))
		# Whole formula dim/2 + dim/2 * np.log(2*np.pi) + 1/2*np.log(det(var))
		# however for comparison reasons omiting the first 2 parts since they will be the same for all
		determinant = det(var)
		if determinant <= 0:
			return 0, var
		else:
			# print(str(determinant) + " " + str(np.log2(determinant)))
			return determinant, var

	def update_stream_greedy(self, selected_users, selected_articles, selected_clicks):
		# print("Update greedy")
		cur_articles_count = len(self.articles) 
		selected_var = []
		max_entropy = -1
		
		if cur_articles_count < self.dimention:
			selected_article = -1
			selected_user = -1
			for id in range(0, len(selected_articles)):
				article_id = selected_articles[id]
				potential_articles = np.append(self.articles, article_id) 
				potential_users = np.append(self.users, selected_users[id])
				entropy, cur_var = self.calculate_stripped_entropy(potential_users, potential_articles)
				if entropy < max_entropy:
					max_entropy = entropy
					selected_article = article_id
					selected_user = selected_users[id]
					selected_var = cur_var

			self.articles = np.append(self.articles, selected_article) 
			self.users = np.append(self.users, selected_user).reshape(cur_articles_count+1, self.d)
			print(self.K)
			self.K = selected_var
		else:
			selected_id_to_remove = -1
			selected_id_to_add = -1
			for id_to_remove in range(0, len(self.articles)):
				for id_to_add in range(0, len(selected_articles)):
					removed_article_id = self.articles[id_to_remove]
					removed_user_id = self.users[id_to_remove]
					self.articles[id_to_remove] = selected_articles[id_to_add]
					self.users[id_to_remove] = selected_users[id_to_add]
					entropy, cur_var = self.calculate_stripped_entropy(self.users, self.articles)
					# print(entropy)
					if entropy > max_entropy:
						# print(entropy)
						max_entropy = entropy
						selected_id_to_add = id_to_add
						selected_id_to_remove = id_to_remove
						selected_var = cur_var
					self.articles[id_to_remove] = removed_article_id
					self.users[id_to_remove] = removed_user_id

			self.articles[selected_id_to_remove] = selected_articles[selected_id_to_add]
			self.users[selected_id_to_remove] = selected_users[selected_id_to_add]
			self.K = selected_var
			print(max_entropy)
		try :
			self.K_i = inv(self.K)
		except np.linalg.linalg.LinAlgError:
			# print(self.K)
			# value = det(self.K)
			# print(value)
			print(" Error singular value")
