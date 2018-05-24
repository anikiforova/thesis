import random
import numpy as np
import pandas as pd
import statistics as stats

random.seed(9999)

def choose_random(user, articles, preselected):	
	return articles.sample(n=1)['ArticleId'].iloc[0]

def choose_all(user, articles, preselected):	
	return preselected

def choose_ucb(user, articles, preselected):
	groups = articles.groupby(['ArticleId'])
	ucb = groups.mean() + groups.var()
	max_ucb = int(ucb.Click.idxmax())
	if(max_ucb == preselected):
		print(ucb.ix[max_ucb])
	return max_ucb

def choose_article(user, articles, preselected, algorithm_type):
	choices = {"random":choose_random, 
				"all"  :choose_all,
				"ucb"  :choose_ucb}

	return choices[algorithm_type](user, articles, preselected) 

choice = "ucb"
skip_impressions = 100
starting_mean = 0.5

fo = open("..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")
output_ctr = open("..//R6//20090501_" + choice + "_ctr_timeline.csv", "w")
output_ctr.write("Impressions, Clicks\n")

articles = pd.DataFrame(columns=['ArticleId', 'Click'], dtype='int32')

total_impressions = 0
total_clicks = 0
joined_impressions = 0
joined_clicks = 0

for line in fo:
	# line = fo.readline()	
	line = line.split("|")

	no_space_line = line[0].split(" ")
	selected_article = int(no_space_line[1])
	click = no_space_line[2]
	user = line[1].split(" ")[0]
	
	total_impressions += 1
	total_clicks += int(click)

	cur_articles = []
	for i in range(2,len(line)):
		cur_article = int(line[i].split(" ")[0])
		cur_articles.append(cur_article)
		item = articles.loc[articles['ArticleId'] == cur_article]
		if(len(item) == 0):
			 articles = articles.append({'ArticleId': cur_article, 'Click': starting_mean}, ignore_index=True)
			 articles = articles.append({'ArticleId': cur_article, 'Click': starting_mean}, ignore_index=True)
	
	articles_to_use = articles[articles['ArticleId'].isin(cur_articles)]
	chosen_article = int(choose_article(user, articles_to_use, selected_article, choice))

	if(chosen_article == selected_article):
		articles = articles.append({'ArticleId': selected_article, 'Click': int(click)}, ignore_index=True)
		
		joined_impressions +=1
		joined_clicks += int(click)
		
		if joined_impressions % skip_impressions == 0:
			output_ctr.write(str(joined_impressions) + ", " + str(joined_clicks) + "\n")
			output_ctr.flush()


articles.to_csv("..//R6//20090501_" + choice + "_baseline.csv", header=True)

print("Total impressions: " + str(total_impressions))
print("Total clicks: " + str(total_clicks))
print("Joined impressions: " + str(joined_impressions))
print("Joined clicks: " + str(joined_clicks))

output_ctr.close()
fo.close()