import numpy as np
import math
from numpy.linalg import inv

class Random:
	
	def __init__(self, alpha):
		np.random.seed(seed=9999)

	def update(self, user, selected_article, click):
		# do nothing
		pass
		
	def select(self, user, lines, exploit):
		cur_articles = list()
		for line in lines:
			article_id = int(line.split(" ")[0])
			cur_articles.append(article_id)
			
		selected_article = np.random.choice(cur_articles, 1)
		return selected_article





