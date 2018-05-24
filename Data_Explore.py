import pandas as pd

fo = open("..//R6//ydata-fp-td-clicks-v1_0.20090501", "r")

users = pd.DataFrame(columns=['Dim2', 'Dim3', 'Dim4', 'Dim5', 'Dim6', 'Dim1'], dtype='int32')
impressions = 0;

for line in fo:
	line = line.split("|")

	user_info = line[1].split(" ")
	users = users.append({'Dim2':user_info[1],
		'Dim3':user_info[2],
		'Dim4':user_info[3],
		'Dim5':user_info[4],
		'Dim6':user_info[5],
		'Dim1':user_info[6]}, ignore_index=True)

	impressions += 1
	if(impressions % 100 == 0):
		users = users.drop_duplicates();
		break;

users.to_csv("..//R6//20090501_users.csv")

fo.close()
