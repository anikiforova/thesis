import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import random

from AlgoBase import AlgoBase 

class NN(AlgoBase):
	
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, filter_clickers = False, soft_click = False):
		super(NN, self).__init__(user_embeddings, user_ids, filter_clickers, soft_click)
		self.d = dimensions
		
		self.learning_rate = 0.001

		# Network Parameters
		self.n_hidden_1  = 20	# 1st hidden layer of neurons
		# self.n_hidden_2  = 32	# 2nd hidden layer of neurons
		# self.n_hidden_3  = 16	# 3rd hidden layer of neurons
		self.n_input     = self.d	# number of features after LSA
		tf.set_random_seed(7855)
		
		# Layer weights, should change them to see results
		self.weights = {
			'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], dtype=np.float64)),       
			# 'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], dtype=np.float64)),
			# 'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3],dtype=np.float64)),
			'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1], dtype=np.float64))
		}
		self.biases = {
			'b1': tf.Variable(tf.random_normal([self.n_hidden_1], dtype=np.float64)),
			# 'b2': tf.Variable(tf.random_normal([n_hidden_2], dtype=np.float64)),
			'out': tf.Variable(tf.random_normal([1], dtype=np.float64))
		}

		# Tensorflow Graph input
		self.x = tf.placeholder(tf.float64, shape=[None, self.n_input], name="x-data")
		self.y = tf.placeholder(tf.float64, shape=[None, 1], name="y-labels")

		# Construct model
		self.pred = self.neural_net(self.x, self.weights, self.biases)

		# Define loss and optimizer
		self.cost = tf.nn.l2_loss(self.pred - self.y, name="squared_error_cost")
		# self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
		# AdagradOptimizer
		self.optimizer = tf.train.AdagradOptimizer() 
		self.optimize = self.optimizer.minimize(self.cost)

		# Initializing the variables
		self.init = tf.global_variables_initializer()
		self.session = tf.Session()
		self.session.run(self.init)
		self.clickers = set()

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(NN, self).prepareClicks(users, clicks)
		
		click_indexes 		= clicks == 1
		no_click_indexes 	= clicks == 0
		new_users_click 	 = set(users[click_indexes])
		new_users_no_click = set(users[no_click_indexes])
		
		for clicker in new_users_click: self.clickers.add(clicker) 
		for clicker in self.clickers: new_users_no_click.discard(clicker) 

		new_users_click 		= np.array(list(new_users_click))
		new_users_no_click 	= np.array(list(new_users_no_click))

		click_count 		= len(new_users_click)
		no_click_count 		= len(new_users_no_click)
		total 				= click_count + no_click_count

		no_click_sample_size= int(total * 0.5)
		click_sample_size 	= total - no_click_sample_size - click_count
		total_click_size = click_sample_size + click_count
		total = no_click_sample_size + total_click_size

		print("Total click: {0} No Click {1}".format(click_count, no_click_count))
	
		click_sample 	= np.random.choice(np.arange(0, len(self.clickers)), click_sample_size, True)
		no_click_sample = np.random.choice(np.arange(0, no_click_count), no_click_sample_size, False)

		batch_user_click_sample = np.array(list(self.clickers))[click_sample]

		if click_count > 0:
			batch_users_click = np.append(new_users_click, batch_user_click_sample)
		else:
			batch_users_click = batch_user_click_sample

		batch_users_no_click = new_users_no_click[no_click_sample]

		batch_user_ids 	= np.append(batch_users_click, batch_users_no_click)		
		batch_clicks 	= np.append(np.ones(total_click_size), np.zeros(no_click_sample_size)).reshape([total, 1])

		batch_user_embeddings	= np.array([self.user_embeddings[id] for id in batch_user_ids]).reshape([total, self.d])
		# total = len(clicks)
		# batch_user_embeddings	= np.array([self.user_embeddings[id] for id in users]).reshape([total, self.d])
		# batch_clicks = clicks.reshape([total, 1])

		_, c = self.session.run([self.optimize, self.cost], feed_dict={self.x: batch_user_embeddings, self.y: batch_clicks})
		train_prediction = self.session.run(self.pred, feed_dict={self.x: batch_user_embeddings})
		
		avg_cost = c / float(total)
		print("Training error=", "{:.9f}".format(avg_cost))
		# print("Training error recalculated =", "{:.9f}".format( calculated_cost))
	
		print("Done with train...")
		# W_val, b_val = self.session.run([self.weights['out'], self.biases['out']])

		self.prediction = self.session.run(self.pred, feed_dict={self.x: self.user_embeddings})
		self.prediction = np.array([item for sublist in self.prediction for item in sublist])
		
		super(NN, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

	def multilayer_perceptron(self, x, weights, biases):
	    # First hidden layer with SIGMOID activation
	    # layer_1 = tf.matmul(x, weights['h1'])
	    # layer_1 = tf.nn.sigmoid(layer_1)
	    # Output layer with SIGMOID activation
	    out_layer = tf.matmul(x, weights['out']) #+ weights['bias']
	    out_layer = tf.nn.sigmoid(out_layer)
	    return out_layer

	def neural_net(self, x, weights, biases):
	    # Hidden fully connected layer with 256 neurons
	    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	    layer_1 = tf.nn.sigmoid(layer_1)
	    # # Hidden fully connected layer with 256 neurons
	    # layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	    # # Output fully connected layer with a neuron for each class
	    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
	    out_layer = tf.nn.sigmoid(out_layer)
	    return out_layer
	

