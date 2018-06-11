from random import randint 
import re
import numpy as np

def to_vector(input):
	input = re.split("[: ]", input)

	return np.array([float(input[2]), float(input[4]), float(input[6]), float(input[8]), float(input[10]), float(input[12])])

fo = open("..//..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")
output = open("..//..//R6//ydata-fp-td-clicks-v1_0.20090501_single_article.csv", "w")

# skip some lines
for i in range (0, 100000):
	fo.readline()

line = fo.readline()
line = line.split("|")
articles = list()
for article in line[2:]:
	article_id = int(article.split(" ")[0])
	articles.append(article_id)

selected_article = articles[randint(0, len(articles)-1)]
count = 0
for line in fo:
	line = line.split("|")
	no_space_line = line[0].split(" ")
	pre_selected_article = int(no_space_line[1])
	
	if(pre_selected_article != selected_article):
		continue

	click = no_space_line[2]
	user = to_vector(line[1])
	user_str = np.array2string(user, separator=' ')[1:-1].replace('\n', '')
	output.write("{0},{1}\n".format(user_str, click))
	count += 1
	if count % 100 == 0:
		output.flush()
		print('.', end='', flush=True)

print(count)


output.close()
fo.close()
		
			