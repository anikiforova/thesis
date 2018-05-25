import numpy as np
import math

class EFirst:
	
	def __init__(self, alpha):
		self.total_lines = 4681992.0
		self.lines_to_explore = self.total_lines * alpha

		self.articles_mean = dict()
		self.articles_clicks = dict()

	def update(self, user, selected_article, click):
		# print(type(selected_article))
		self.articles_clicks[selected_article].append(click)
		self.articles_mean[selected_article] = np.mean(self.articles_clicks[selected_article])
		
	def warmup(self, fo):
		pass
		
	def select(self, user, lines, total_impressions):
		explore_articles = list()

		best_mean = 0
		selected_article = -1

		for line in lines:
			article_id = int(line.split(" ")[0])
			if article_id in self.articles_clicks.keys():	
				if len(self.articles_clicks[article_id]) < self.lines_to_explore:
					explore_articles.append(article_id)
				else:
					cur_mean = self.articles_mean[article_id]
					if best_mean < cur_mean:
						best_mean = cur_mean
						selected_article = article_id
			else:
				self.articles_clicks[article_id] = list()
				self.articles_mean[article_id] = 0

		if len(explore_articles) > 0:
			selected_article = np.random.choice(explore_articles, 1)

		if len(explore_articles) == 1:
			print("Found a new article to explore")
		return selected_article

