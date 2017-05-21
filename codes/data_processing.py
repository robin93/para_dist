import pandas as pd, numpy as np
import urllib2
import ast
import csv

def main():
	# convert_extrainfo_json_to_table()
	# create_validation_set()
	# cal_aggregate_user_stats()
	get_deezer_api_data()

def get_deezer_api_data():
	# frBTF0DSipIhniw3iv5PU5rbYT4CtRpGNb1MD4hb539MOVpynK2
	# secret key 6352f569e868cde70681a49755d65561
	# application domain http://xyz.com
	# https://connect.deezer.com/oauth/auth.php?app_id=233182&redirect_uri=http://xyz.com&perms=basic_access
	X_train = pd.read_csv("../data/train.csv")
	pivot = pd.DataFrame(pd.pivot_table(X_train,index="media_id",values="is_listened",aggfunc=len))
	subset = pivot[pivot.is_listened>10]
	subset.to_csv("../files/more_than_10_list_songs.csv")
	



	def user_data():
		user_series = X_train["user_id"].unique()
		cnt = 0
		output= open("../files/user_api_data.csv","w")
		for user in user_series:
			cnt += 1
			if cnt%100==0:
				print cnt,"-",
			# if cnt > 5:
		# 	# 	break
			url = "http://api.deezer.com/user/"+str(user)
			html = eval(urllib2.urlopen(url).read())
			try:
				user_id,name,country,user_type = html["id"],html["name"],html["country"],html["type"]
				to_write = str(cnt)+","+str(user_id)+","+str(name)+","+str(country)+","+str(user_type)+"\n"
				print to_write
				output.writelines(to_write)
			except Exception:
				continue


	def song_data():
		song_series = X_train["media_id"].unique()
		with open("../files/song_api_data.csv","a") as f:
			writer = csv.writer(f)
			cnt = 0
			for song in song_series:
				cnt += 1
				url = "http://api.deezer.com/track/"+str(song)
				dictionary = eval(urllib2.urlopen(url).read().replace("true","\"T\"").replace("false","\"F\""))
				try:
					song_id = dictionary["id"]
					readable = dictionary["readable"]
					explicit = dictionary["explicit_lyrics"]
					bpm = dictionary["bpm"]
					number_of_countries_availaible = len(dictionary["available_countries"])
					gain = dictionary["gain"]
					to_write = str(song_id)+","+str(readable)+","+str(explicit)+","+str(bpm)+","+str(number_of_countries_availaible)+","+str(gain)+"\n"
					print cnt,to_write
					# output.write(to_write)
					writer.writerow([song_id,readable,explicit,bpm,number_of_countries_availaible,gain])
				except Exception:
					continue
		 
	def artist_data():
		artist_series = X_train["artist_id"].unique()
		with open("../files/artist_api_data.csv","a") as f:
			writer = csv.writer(f)
			cnt = 0
			for artist in artist_series:
				cnt += 1
				url = "http://api.deezer.com/artist/"+str(artist)
				dictionary = eval(urllib2.urlopen(url).read().replace("true","\"T\"").replace("false","\"F\""))
				try:
					artist = dictionary["id"]
					nb_album = dictionary["nb_album"]
					ab_fan = dictionary["nb_fan"]
					radio = dictionary["radio"]
					to_write = str(artist)+","+str(nb_album)+","+str(ab_fan)+","+str(radio)
					print cnt,to_write
					# output.write(to_write)
					writer.writerow([artist,nb_album,ab_fan,radio])
				except Exception:
					continue
	# artist_data()
	# song_data()
	# user_data()
			




def cal_aggregate_user_stats():
	X_train = pd.read_csv("../data/train.csv",nrows=1000)
	unique_user_list = X_train["user_id"].unique()
	for user in unique_user_list:	
		print user,np.mean(X_train[X_train.user_id==user]["is_listened"])


def convert_extrainfo_json_to_table():
	import json 
	cnt = 0
	key_list = ['media_id','sng_title','alb_title','art_name']
	output = open("files/extra_info_table.csv","w")
	output.writelines("media_id,sng_title,alb_title,art_name\n")
	with open("data/extra_infos.json") as datafile:
		for line in datafile:
			cnt += 1
			# if cnt>5:
			# 	break
			info_dict = dict(eval(line.strip()))
			to_write = ""
			newline_check = 0
			for key in key_list:
				newline_check += 1
				if newline_check == 4:
					to_write = to_write + str(info_dict[key]) + "\n"
				else:
					to_write = to_write + str(info_dict[key]) + ","
			output.writelines(to_write)

def create_validation_set():
	import pandas as pd
	import numpy as np
	# X_train = pd.read_csv("../data/train.csv")
	X_train = pd.read_csv("../data/with_transition_prob/TrainWithGlobalTransitionProbabilities.csv")
	del X_train["Unnamed: 0"]
	#get unique list of users
	list_of_users = X_train["user_id"].unique()
	print len(list_of_users)


	pd.DataFrame(columns=X_train.columns).to_csv("../data/trainval_globaltrans_val.csv",index=False)
	pd.DataFrame(columns=X_train.columns).to_csv("../data/trainval_globaltrans_train.csv",index=False)


	#subset the data by user
	cnt = 0
	for user in list_of_users:
		user_data = X_train[(X_train.user_id==user)]
		if len(user_data)>2:
			max_ts = np.max(user_data['ts_listen'].ravel())
			# listen_time_array = user_data['ts_listen'].ravel()
			# second_highest = listen_time_array.sort()[-2]
			val_df = user_data[(user_data.ts_listen==max_ts)]
			train_df = user_data[(user_data.ts_listen<max_ts)]
			val_df.to_csv("../data/trainval_globaltrans_val.csv",mode='a',header=False,index=False)
			train_df.to_csv("../data/trainval_globaltrans_train.csv",mode='a',header=False,index=False)
			cnt += 1
			# if cnt>10:
			# 	break
			if cnt%1000==0:
				print cnt

if __name__ == '__main__':
    main()



