import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import random

from Util import to_vector

class NN:
	
	def __init__(self, alpha):
		self.alpha = alpha
		self.d = 36	
		self.actions = 2 # 0, 1 
		self.learning_rate = 0.001

		self.epoch = 0
		self.display_step    =  50

		self.mini_batch_size = 1000
		self.batch_a_clicks = np.array([])
		self.batch_a_no_clicks = np.array([])
		self.batch_click_count = 0
		self.batch_no_click_count = 0

		self.articles = dict()
		self.bad_articles = set()
		# Network Parameters

		self.n_hidden_1  = 10	# 1st hidden layer of neurons
		self.n_hidden_2  = 32	# 2nd hidden layer of neurons
		self.n_hidden_3  = 16	# 3rd hidden layer of neurons
		self.n_input     = 36	# number of features after LSA
		
		# Layer weights, should change them to see results
		# self.weights = {
		# 	'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], dtype=np.float64)),       
		# 	'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], dtype=np.float64)),
		# 	'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3],dtype=np.float64)),
		# 	'out': tf.Variable(tf.random_normal([self.n_hidden_3, 1], dtype=np.float64))
		# }

		self.weights = {
			'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], dtype=np.float64)),       
			# 'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], dtype=np.float64)),
			# 'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3],dtype=np.float64)),
			'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1], dtype=np.float64))
		}

		# self.choose_action = tf.argmax(self.weights, 0)

		# Tensorflow Graph input
		self.x = tf.placeholder(tf.float64, shape=[None, self.n_input], name="x-data")
		self.y = tf.placeholder(tf.float64, shape=[None, 1], name="y-labels")

		# Construct model
		self.pred = self.multilayer_perceptron(self.x, self.weights)

		# Define loss and optimizer
		self.cost = tf.nn.l2_loss(self.pred - self.y, name="squared_error_cost")
		self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

		# Initializing the variables
		self.init = tf.global_variables_initializer()
		self.session = tf.Session()
		self.session.run(self.init)

	def add_new_article(self, line):
		article_id = int(line.split(" ")[0])
			
		if article_id in self.bad_articles:
			return -1

		if article_id not in self.articles:
			try:
				article = to_vector(line)
			except IndexError:
				print("Skipping line, weird formatting.." + str(article_id))
				self.bad_articles.add(article_id)
				return -1

			self.articles[article_id] = article.reshape([1, 6])
		return article_id

	def update(self, user, selected_article, click):
		pair = user.reshape([6,1]).dot(self.articles[selected_article])

		if click:
			self.batch_a_clicks = np.append(self.batch_a_clicks, pair)
			self.batch_click_count+=1
		else:
			self.batch_a_no_clicks = np.append(self.batch_a_no_clicks, pair)
			self.batch_no_click_count += 1

		total = self.batch_click_count + self.batch_no_click_count
		
		if total % self.mini_batch_size == 0 and self.batch_click_count > 0:
			sample_size = int(total / 2)
			total = int(sample_size * 2)

			sample_clicks 			= np.random.choice(np.arange(0, self.batch_click_count), sample_size, True)
			sample_no_clicks 		= np.random.choice(np.arange(0, self.batch_no_click_count), sample_size, False)
			
			self.batch_a_clicks 	= self.batch_a_clicks.reshape([self.batch_click_count, self.d])[sample_clicks]
			self.batch_a_no_clicks 	= self.batch_a_no_clicks.reshape([self.batch_no_click_count, self.d])[sample_no_clicks]
				
			articles = np.append(self.batch_a_clicks, self.batch_a_no_clicks).reshape([ total, self.d])
			clicks = np.append(np.ones(sample_size), np.zeros(sample_size)).reshape([ total, 1])

			_, c = self.session.run([self.optimizer, self.cost], feed_dict={self.x: articles, self.y: clicks})
			avg_cost = c / (total)

			print("Training error=", "{:.9f}".format(avg_cost))
			self.batch_a_clicks 		= np.array([])
			self.batch_a_no_clicks 		= np.array([])
			self.batch_click_count 		= 0	
			self.batch_no_click_count 	= 0

		# if click :
		# 	self.batch_a_clicks = np.append(self.batch_a_clicks, pair)
		# 	self.batch_clicks = np.append(self.batch_clicks, click)
		# else: 
		# 	self.batch_a_no_clicks = np.append(self.batch_a_no_clicks, pair)
		# 	self.batch_no_clicks = np.append(self.batch_no_clicks, click)

		# click_count = len(self.batch_clicks)
		# no_click_count = len(self.batch_no_clicks)	
		# # print(str(click_count) + " " + str(no_click_count) + " " + str(click_count +no_click_count))
		# if (click_count + no_click_count) % self.mini_batch_size == 0 and click_count > 0:	
		# 	# print("Trainging 	")
		# 	click_sample_size = int(no_click_count * 0.2)
		# 	sample_size = no_click_count + click_sample_size
		# 	sample = np.random.choice(np.arange(0, click_count), click_sample_size, True)
		# 	self.batch_a_clicks = self.batch_a_clicks.reshape([click_count, self.d])[sample]
		# 	self.batch_clicks = self.batch_clicks[sample]

		# 	articles = np.append(self.batch_a_clicks, self.batch_a_no_clicks).reshape([sample_size, self.d])
		# 	clicks = np.append(self.batch_clicks, self.batch_no_clicks).reshape([sample_size, 1])
			
		# 	_, c = self.session.run([self.optimizer, self.cost], feed_dict={self.x: articles, self.y: clicks})

		# 	self.batch_a_clicks = np.array([])
		# 	self.batch_clicks = np.array([])
		# 	self.batch_a_no_clicks = np.array([])
		# 	self.batch_no_clicks = np.array([])

		# 	# Display logs per epoch
		# 	# if self.epoch % self.display_step == 0:
		# 	avg_cost = c / (2 * click_count)
		# 	print("Epoch:", '%05d' % self.epoch, "Training error=", "{:.9f}".format(avg_cost))
		# 	self.epoch += 1

	def warmup(self, fo):
		pass

	def select(self, user, pre_selected_article, lines, total_impressions, click):
		bucket = random.uniform(0, 1)
		explore = bucket <= self.alpha
		# print(EGreedy.alpha)
		cur_articles = list()
		best_action = -1000
		selected_article = -1
		best_articles = list()

		if explore:
			for line in lines:
				article_id = self.add_new_article(line)	
				cur_articles.append(article_id)

			selected_article = np.random.choice(cur_articles, 1)
		else:
			user = user.reshape([6, 1])
			for line in lines:
				article_id = self.add_new_article(line)			
				if article_id == -1:
					continue
				pair = user.dot(self.articles[article_id].reshape([1, 6])).reshape([1, 36])
				action = self.session.run(self.pred, feed_dict={self.x: pair})[0][0]
				# print(action)

				if best_action < action:
					best_action = action
					best_articles = list([article_id])

				elif best_action == action:
					best_articles.append(article_id)
			
			selected_article = np.random.choice(best_articles, 1)
			
		return selected_article, False

	# Create model
	# def multilayer_perceptron(self, x, weights):
	#     # First hidden layer with SIGMOID activation
	#     layer_1 = tf.matmul(x, weights['h1'])
	#     layer_1 = tf.nn.sigmoid(layer_1)
	#     # Second hidden layer with SIGMOID activation
	#     layer_2 = tf.matmul(layer_1, weights['h2'])
	#     layer_2 = tf.nn.sigmoid(layer_2)
	#     # Third hidden layer with SIGMOID activation
	#     layer_3 = tf.matmul(layer_2, weights['h3'])
	#     layer_3 = tf.nn.sigmoid(layer_3)
	#     # Output layer with SIGMOID activation
	#     out_layer = tf.matmul(layer_3, weights['out'])
	#     out_layer = tf.nn.sigmoid(out_layer)
	#     return out_layer

	def multilayer_perceptron(self, x, weights):
	    # First hidden layer with SIGMOID activation
	    layer_1 = tf.matmul(x, weights['h1'])
	    layer_1 = tf.nn.sigmoid(layer_1)
	    # # Output layer with SIGMOID activation
	    out_layer = tf.matmul(layer_1, weights['out'])
	    out_layer = tf.nn.sigmoid(out_layer)
	    return out_layer
