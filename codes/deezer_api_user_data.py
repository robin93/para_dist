import pandas as pd, numpy as np
import urllib2

X_train = pd.read_csv("../data/train.csv")
	user_series = X_train["user_id"].unique()
	cnt = 0
	output= open("../files/user_api_data.csv","w")
	for user in user_series:
		cnt += 1
		if cnt%100==0:
			print cnt,"-",
		# if cnt > 5:
		# 	break
		url = "http://api.deezer.com/user/"+str(user)
		html = eval(urllib2.urlopen(url).read())
		try:
			user_id,name,country,user_type = html["id"],html["name"],html["country"],html["type"]
			to_write = str(cnt)+","+str(user_id)+","+str(name)+","+str(country)+","+str(user_type)+"\n"
			output.writelines(to_write)
		except Exception:
			continue
		