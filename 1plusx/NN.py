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
		self.n_hidden_1  = 10	# 1st hidden layer of neurons
		# self.n_hidden_2  = 32	# 2nd hidden layer of neurons
		# self.n_hidden_3  = 16	# 3rd hidden layer of neurons
		self.n_input     = self.d	# number of features after LSA
		
		# Layer weights, should change them to see results
		self.weights = {
			'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], dtype=np.float64)),       
			# 'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], dtype=np.float64)),
			# 'h3': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_3],dtype=np.float64)),
			'out': tf.Variable(tf.random_normal([self.n_hidden_1, 1], dtype=np.float64))#,
			#'bias': tf.Variable((1, ), tf.constant_initializer(0.0, dtype=tf.float64))
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

	def update(self, users, clicks):
		print("Starting Update.. ", end='', flush=True)
		users, clicks = super(NN, self).prepareClicks(users, clicks)
		
		click_indexes 		= clicks == 1
		no_click_indexes 	= clicks == 0

		batch_users_click 	 = users[click_indexes]
		batch_users_no_click = users[no_click_indexes]
		
		total 			= len(clicks)		
		sample_size 	= int(total / 2)
		click_count 	= sum(click_indexes)
		no_click_count 	= sum(no_click_indexes)

		click_sample 	= np.random.choice(np.arange(0, click_count), 	 sample_size, True)
		no_click_sample = np.random.choice(np.arange(0, no_click_count), sample_size, False)

		batch_users_click 		= batch_users_click[click_sample]
		batch_users_no_click 	= batch_users_no_click[no_click_sample]

		batch_user_ids 			= np.append(batch_users_click, batch_users_no_click)		
		batch_clicks 			= np.append(np.ones(sample_size), np.zeros(sample_size))
		batch_user_embeddings	= np.array([self.user_embeddings[id] for id in batch_user_ids]).reshape([total, self.d])

		_, c = self.session.run([self.optimizer, self.loss], feed_dict={self.x: batch_user_embeddings, self.y: batch_clicks})
		avg_cost = c / (2 * click_count)
		print("Training error=", "{:.9f}".format(avg_cost))
	
		print("Done with train...")
		self.prediction = self.session.run(self.y_pred, feed_dict={self.x: self.user_embeddings})
		print("Done with prediction...")
		
		super(NN, self).predictionPosprocessing(users, clicks)		
		print(" Done.")

	def multilayer_perceptron(self, x, weights):
	    # First hidden layer with SIGMOID activation
	    layer_1 = tf.matmul(x, weights['h1'])
	    layer_1 = tf.nn.sigmoid(layer_1)
	    # Output layer with SIGMOID activation
	    out_layer = tf.matmul(layer_1, weights['out']) #+ weights['bias']
	    out_layer = tf.nn.sigmoid(out_layer)
	    return out_layer
	

