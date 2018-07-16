import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import random

from AlgoBase import AlgoBase 

class NN(AlgoBase):
	
	def __init__(self, user_embeddings, user_ids, cluster_embeddings, dimensions, click_percent = 0.5, equalize_clicks = False, filter_clickers = False, soft_click = False):
		super(NN, self).__init__(user_embeddings, user_ids, click_percent, equalize_clicks, filter_clickers, soft_click)
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
		# self.optimizer = tf.train.AdagradOptimizer(self.learning_rate) 
		self.optimizer = tf.train.AdamOptimizer() 
		self.optimize = self.optimizer.minimize(self.cost)

		# Initializing the variables
		self.init = tf.global_variables_initializer()
		self.session = tf.Session()
		self.session.run(self.init)

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(NN, self).prepareClicks(users, clicks)
		
		total = len(clicks)
		batch_user_embeddings	= np.array([self.user_embeddings[id] for id in users]).reshape([total, self.d])
		
		_, c = self.session.run([self.optimize, self.cost], feed_dict={self.x: batch_user_embeddings, self.y: clicks})
		avg_cost = c / float(total)
		print("Training error=", "{:.9f}".format(avg_cost))
			
		print("Done with train...")
		self.prediction = self.session.run(self.pred, feed_dict={self.x: self.user_embeddings})
		self.prediction = np.array([item for sublist in self.prediction for item in sublist])
		
		super(NN, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

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
	

