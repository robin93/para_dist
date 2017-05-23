import pandas as pd,numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt


def features_to_add(X_train,X_test,validation):
	#add_is_listened_avg(X_train,X_test,validation)
	# add_is_listened_on_flow_avg(X_train,X_test)
	#one_hot_encoding_context(X_train,X_test)
	#add_artist_api_data(X_train,X_test)
        #song_api_data(X_train,X_test)
        #song_age(X_train,X_test)
	# user_age_song_age(X_train,X_test)
	#add_song_is_listened_avg(X_train,X_test,validation)
	# add_genre_is_listened_avg(X_train,X_test,validation=is_validation)
	#X_train["platform_family_feat"] = X_train["platform_family"].apply(lambda x: 1 if x in [1,2] else 0)
	#X_test["platform_family_feat"] = X_test["platform_family"].apply(lambda x: 1 if x in [1,2] else 0)
	#X_train["platform_type_0"] = X_train["platform_name"].apply(lambda x: 1 if x==0 else 0)
	#X_test["platform_type_0"] = X_test["platform_name"].apply(lambda x: 1 if x==0 else 0)
	#X_train["platform_type_1"] = X_train["platform_name"].apply(lambda x: 1 if x==1 else 0)
	#X_test["platform_type_1"] = X_test["platform_name"].apply(lambda x: 1 if x==1 else 0)
	#X_train["platform_type_2"] = X_train["platform_name"].apply(lambda x: 1 if x==2 else 0)
	#X_test["platform_type_2"] = X_test["platform_name"].apply(lambda x: 1 if x==2 else 0)
	# avg_time_user_listen_song(X_train,X_test)
	#X_train["high_listen_hour"] = X_train["ts_listen"].apply(lambda x: 1 if datetime.fromtimestamp(x).hour in [7,8,9,10,14,15,16,20] else 0)
	#X_test["high_listen_hour"] = X_test["ts_listen"].apply(lambda x: 1 if datetime.fromtimestamp(x).hour in [7,8,9,10,14,15,16,20] else 0)
	# add_user_total_len(X_train,X_test)
	# first_time_on_flow(X_train,X_test)
	#number_of_rows_for_user(X_train,X_test)
	# user_time_to_test_listen(X_train,X_test)
	# user_listen_changes_factor(X_train,X_test)
	#add_artist_api_data(X_train,X_test)
	#song_api_data(X_train,X_test)
	# previous_is_listened(X_train,X_test)
        #one_hot_encoding_context(X_train,X_test)
        user_country_assignment(X_train,X_test)

def user_country_assignment(X_train,X_test):
    with open("../files/user_country_dict.json") as datafile:
        user_country_dict = json.load(datafile)
    user_set = set(user_country_dict.keys())
    print "writing for france"
    X_train["is_france"] = X_train["user_id"].apply(lambda x: 1 if str(x) in user_set and user_country_dict[str(x)]=="FR" else 0)
    X_test["is_france"] = X_test["user_id"].apply(lambda x: 1 if str(x) in user_set and user_country_dict[str(x)]=="FR" else 0)
    print "writing for US"
    X_train["is_US"] = X_train["user_id"].apply(lambda x: 1 if str(x) in user_set and user_country_dict[str(x)]=="US" else 0)
    X_test["is_US"] = X_test["user_id"].apply(lambda x: 1 if str(x) in user_set and user_country_dict[str(x)]=="US" else 0)
    print "writing for GB"
    X_train["is_GB"] = X_train["user_id"].apply(lambda x: 1 if str(x) in user_set and user_country_dict[str(x)]=="GB" else 0)        
    X_test["is_GB"] = X_test["user_id"].apply(lambda x: 1 if str(x) in user_set and user_country_dict[str(x)]=="GB" else 0)
        

# X_train = pd.read_csv("../data/trainval_train.csv",nrows=1000)
# X_train = pd.read_csv("../data/train.csv")
# # # subset = X_train[X_train.user_id==258]
# # # subset.to_csv("../files/258_user_data.csv",index=False)
# # # X_test = pd.read_csv("../data/test.csv")
# # # print X_test
# pivot = pd.DataFrame(pd.pivot_table(X_train,values="is_listened",index="user_id",columns="listen_type",aggfunc=[len,np.sum]))
# pivot.to_csv("../files/listen_type_is_listened.csv")
# # print X_train.sort(["user_id","ts_listen"])[["user_id","ts_listen"]].head(100)

# %matplotlib inline

def by_date_is_listened_avg():
	X_train = pd.read_csv("../data/trainval_train.csv",nrows=1000)
	X_test = pd.read_csv("../data/test.csv")
	X_train["date"] = X_train["ts_listen"].apply(lambda x: datetime.fromtimestamp(x).date())
	X_test["date"] = X_test["ts_listen"].apply(lambda x: datetime.fromtimestamp(x).date())
	pivot = pd.pivot_table(X_train,values="is_listened",index="date",aggfunc=np.mean)
	date_list,avg_values = pivot.index.values,list(pivot)
	print date_list
	print avg_values

# by_date_is_listened_avg()





def top_streaming_artist(X_train,X_test):
	# first_date = datetime.fromtimestamp(np.min(X_train["ts_listen"])).date()
	# last_date = datetime.fromtimestamp(np.max(X_train["ts_listen"])).date()
	# print first_date,last_date
	X_train["date"] = X_train["ts_listen"].apply(lambda x: datetime.fromtimestamp(x).date())
	X_test["date"] = X_test["ts_listen"].apply(lambda x: datetime.fromtimestamp(x).date())
	pivot = pd.pivot_table(X_test,values="user_id",index="date",aggfunc=len)
	print pivot
	pivot_train = pd.pivot_table(X_train,values="is_listened",index="date",aggfunc=[len,np.mean])
	pivot_train.to_csv("../files/date_variations.csv")

# X_train = pd.read_csv("../data/trainval_train.csv")
# X_test = pd.read_csv("../data/test.csv")
# top_streaming_artist(X_train,X_test)

def by_date_average(X_train,X_test):
	print "hello world"

def media_duration_outlier(X_train,X_test):
	X_train["media_dur_high_outlier"] = X_train["media_duration"].apply(lambda x: 1 if int(x)>1000 else 0)
	X_test["media_dur_high_outlier"] = X_test["media_duration"].apply(lambda x: 1 if int(x)>1000 else 0)
	X_train["media_dur_low_outlier"] = X_train["media_duration"].apply(lambda x: 1 if int(x)<50 else 0)
	X_test["media_dur_low_outlier"] = X_test["media_duration"].apply(lambda x: 1 if int(x)<50 else 0)

def user_country():
	print "work in progress"

def user_pref_of_choice_flow(X_train,X_test):
	# with open("../files/user_both_type_listen_avg_crossval.json") as datafile:
	# 		user_both_type_avg_dict = json.load(datafile)
	# list_of_differences = list()
	# prefers_choice,prefers_flow = list(),list()
	# for user in user_both_type_avg_dict.keys():
	# 	if int(user)<17000 and int(user)>30:
	# 		diff  = float(user_both_type_avg_dict[user][0]) - float(user_both_type_avg_dict[user][1])
	# 		if diff > 0.5:
	# 			prefers_choice.append(user)
	# 		if diff < -0.1:
	# 			prefers_flow.append(user)

	# with open("../files/users_prefering_choice.json","w") as outfile:
	# 	json.dump(prefers_choice,outfile)
	# with open("../files/users_prefering_flow.json","w") as outfile:
	# 	json.dump(prefers_flow,outfile)
	# list_of_differences.append(diff)
	# print np.mean(list_of_differences)
	# print np.std(list_of_differences)
	# plt.hist(list_of_differences, normed=False, bins=100)
	# plt.show()
	# user_len_list = list()
	# with open("../files/user_len_dict.json") as datafile:
	# 	user_len_dict = json.load(datafile)
	# for user in user_len_dict.keys():
	# 	user_len_list.append(user_len_dict[user])
	# plt.hist(user_len_list,bins=100)
	# plt.show()
	with open("../files/users_prefering_choice.json") as datafile:
		prefer_choice_list = set(json.load(datafile))
	with open("../files/users_prefering_flow.json") as datafile:
		prefers_flow_list = set(json.load(datafile))
	X_train["prefer_choice"] = X_train["user_id"].apply(lambda x: 1 if str(x) in prefer_choice_list else 0)
	X_test["prefer_choice"] = X_test["user_id"].apply(lambda x: 1 if str(x) in prefer_choice_list else 0)
	X_train["prefer_flow"] = X_train["user_id"].apply(lambda x: 1 if str(x) in prefers_flow_list else 0)
	X_test["prefer_flow"] = X_test["user_id"].apply(lambda x: 1 if str(x) in prefers_flow_list else 0)

def song_api_data(X_train,X_test):
	# X_test = pd.read_csv("../data/trainval_val.csv")
	# test_song_list = set(X_test["media_id"].unique())
	# song_data_dict = dict()
	# with open("../files/all_songs_api_data.csv") as datafile:
	# 	cnt = 0
	# 	for line in datafile:
	# 		cnt += 1
	# 		# if cnt > 100000:
	# 		# 	break
	# 		song_data = line.strip().split(",")
	# 		media_id = song_data[0]
	# 		if int(media_id) in test_song_list:
	# 			song_readable = 1 if song_data[1]=="T" else 0
	# 			song_explicit = 1 if song_data[2]=="T" else 0
	# 			song_bpm = int(float(song_data[3]))
	# 			gain = int(float(song_data[4]))
	# 			song_data_dict[int(media_id)] = [song_readable,song_explicit,song_bpm,gain]
	# with open("../files/song_data_dict.json","w") as outfile:
	# 	json.dump(song_data_dict,outfile)
	def song_is_readable_assign(row):
		song = row["media_id"]
		try:
			return song_data_dict[unicode(str(int(song)),"utf-8")][0]
		except:
			return 0

	def song_is_explicit_assign(row):
		song = row["media_id"]
		try:
			return song_data_dict[unicode(str(int(song)),"utf-8")][1]
		except:
			return 0

	def song_bpm_assign(row):
		song = row["media_id"]
		try:
			return song_data_dict[unicode(str(int(song)),"utf-8")][2]
		except:
			return 0
	
	def song_gain_assign(row):
		song = row["media_id"]
		try:
			return song_data_dict[unicode(str(int(song)),"utf-8")][3]
		except:
			return 0


	with open("../files/song_data_dict.json") as datafile:
		song_data_dict = json.load(datafile)
	X_train["song_is_readable"] = X_train.apply(song_is_readable_assign,axis=1)
	X_test["song_is_readable"] = X_test.apply(song_is_readable_assign,axis=1)
	X_train["song_is_explicit"] = X_train.apply(song_is_explicit_assign,axis=1)
	X_test["song_is_explicit"] = X_test.apply(song_is_explicit_assign,axis=1)
	X_train["song_bpm"] = X_train.apply(song_bpm_assign,axis=1)
	X_test["song_bpm"] = X_test.apply(song_bpm_assign,axis=1)
	X_train["song_gain"] = X_train.apply(song_gain_assign,axis=1)
	X_test["song_gain"] = X_test.apply(song_gain_assign,axis=1)

def previous_is_listened(X_train,X_test):
	# user_list = X_train["user_id"].unique()
	# print len(user_list)
	# user_timestamp_is_listened_sorted_dict = dict()
	# cnt = 0
	# for user in user_list:
	# 	cnt += 1
	# 	# if cnt%100==0:
	# 	print cnt
	# 	subset = X_train[X_train.user_id==user][["ts_listen","is_listened"]].sort_values(by="ts_listen")
	# 	ts_listen_list = list(subset["ts_listen"])
	# 	is_listened_list = list(subset["is_listened"])
	# 	user_timestamp_is_listened_sorted_dict[user] = [ts_listen_list,is_listened_list]
	# with open("../files/user_timestamp_is_listened_sorted_dict_all.json","w") as outfile:
	# 	json.dump(user_timestamp_is_listened_sorted_dict,outfile)
	with open("../files/user_timestamp_is_listened_sorted_dict_all.json") as datafile:
		user_timestamp_is_listened_sorted_dict = json.load(datafile)
	users_in_test = X_test["user_id"].unique()
	def is_listened_previous_assign(row):
		user,timestamp = row["user_id"],row["ts_listen"]
		try:
			required_index = user_timestamp_is_listened_sorted_dict[unicode(str(int(user)),"utf-8")][0].index(timestamp)-1
			if required_index>=0:
				return user_timestamp_is_listened_sorted_dict[unicode(str(int(user)),"utf-8")][1][required_index]
			else:
				# return user_timestamp_is_listened_sorted_dict[unicode(str(int(user)),"utf-8")][1][required_index+1]
				return -1
		except:
			return user_timestamp_is_listened_sorted_dict[unicode(str(int(user)),"utf-8")][1][-1]

	X_train["is_listened_previous"] = X_train.apply(is_listened_previous_assign,axis=1)
	X_test["is_listened_previous"] = X_test.apply(is_listened_previous_assign,axis=1)


# def user_api_data():
# 	user_data = pd.read_csv("../data/user_name_country_api_data.csv")
# 	user_country_dict= dict(zip(user_data["user_id"],user_data["user_country"]))
# 	print len(user_country_dict.keys())
# 	X_train = pd.read_csv("../data/train.csv")
# 	def user_country_assign(row):
# 		user = row["user_id"]
# 		try:
# 			return user_country_dict[user]
# 		except:
# 			return "NA"
# 	X_train["user_country"] = X_train.apply(user_country_assign,axis=1)
# 	pivot = pd.DataFrame(pd.pivot_table(X_train,index="user_country",values="is_listened",aggfunc=[len,np.mean]))
# 	pivot.to_csv("../files/user_country_is_listened_avg.csv")


def add_artist_api_data(X_train,X_test):
	# artist_attributes_dict = dict()
	# with open("../data/artist_api_data_3.csv") as datafile:
	# 	cnt = 0
	# 	for line in datafile:
	# 		artist_data = line.split("&")
	# 		for artist in artist_data:
	# 			cnt += 1
	# 			if cnt %100 == 0:
	# 				print cnt
	# 			# if cnt > 5:
	# 			# 	break
	# 			if cnt > 1:
	# 				artist_att_list = artist.split(",")
	# 				artist_id = artist_att_list[0]
	# 				nb_album = int(artist_att_list[1])
	# 				nb_fans = int(artist_att_list[2])
	# 				nb_fans_per_album = int(artist_att_list[3])
	# 				is_radio = int(artist_att_list[5])
	# 				artist_attributes_dict[artist_id] = [nb_album,nb_fans,nb_fans_per_album,is_radio]
	# with open("../files/artist_attributes_dict.json","w") as outfile:
	# 	json.dump(artist_attributes_dict,outfile)
	def artist_fan_number_assign(row):
		artist_id = row["artist_id"]
		try:
			return artist_attributes_dict[unicode(str(int(artist_id)),"utf-8")][1]
		except:
			return 0

	def artist_album_number_assign(row):
		artist_id = row["artist_id"]
		try:
			return artist_attributes_dict[unicode(str(int(artist_id)),"utf-8")][0]
		except:
			return 0

	def fans_per_album_assign(row):
		artist_id = row["artist_id"]
		try:
			return artist_attributes_dict[unicode(str(int(artist_id)),"utf-8")][2]
		except:
			return 0

	def artist_is_on_radio(row):
		artist_id = row["artist_id"]
		try:
			return int(artist_attributes_dict[unicode(str(int(artist_id)),"utf-8")][3])
		except:
			return 0

	with open("../files/artist_attributes_dict.json") as datafile:
		artist_attributes_dict = json.load(datafile)
	X_train["artist_fan_number"] = X_train.apply(artist_fan_number_assign,axis=1)
	X_test["artist_fan_number"] = X_test.apply(artist_fan_number_assign,axis=1)
	X_train["artist_nb_album"] = X_train.apply(artist_album_number_assign,axis=1)
	X_test["artist_nb_album"] = X_test.apply(artist_album_number_assign,axis=1)
	X_train["artist_fan_per_album"] = X_train.apply(fans_per_album_assign,axis=1)
	X_test["artist_fan_per_album"] = X_test.apply(fans_per_album_assign,axis=1)
	X_train["artist_is_on_radio"] = X_train.apply(artist_is_on_radio,axis=1)
	X_test["artist_is_on_radio"] = X_test.apply(artist_is_on_radio,axis=1)

def user_listen_changes_factor(X_train,X_test):
	# X_train = pd.read_csv("../data/trainval_train.csv")
	# user_list = X_train["user_id"].unique()
	# user_listen_change_dict = dict()
	# cnt = 0
	# for user in user_list:
	# 	try:
	# 		is_listened_pattern = X_train[X_train.user_id==user][["ts_listen","is_listened"]].sort("ts_listen")["is_listened"].ravel()
	# 		changes_count = 0
	# 		one_count = float(np.sum(is_listened_pattern))
	# 		current = is_listened_pattern[0]
	# 		cnt += 1
	# 		if cnt%100==0:
	# 			print cnt
	# 		for item in is_listened_pattern:
	# 			if item != current:
	# 				changes_count+= 1
	# 				current = item
	# 		user_listen_change_dict[user] = (changes_count/one_count)
	# 	except:
	# 		user_listen_change_dict[user] = 0
	# with open("../files/user_is_listened_change_dict.json","w") as outfile:
	# 	json.dump(user_listen_change_dict,outfile)
	def user_listen_change_assign(row):
		user = row["user_id"]
		try:
			return user_listen_change_dict[unicode(str(int(user)),"utf-8")]
		except:
			return 0
	with open("../files/user_is_listened_change_dict.json") as datafile:
		user_listen_change_dict = json.load(datafile)
	X_train["user_listen_variance"] = X_train.apply(user_listen_change_assign,axis=1)
	X_test["user_listen_variance"] = X_test.apply(user_listen_change_assign,axis=1)

def user_time_to_test_listen(X_train,X_test):
	# X_test = pd.read_csv("../data/test.csv")
	# user_test_listen_dict = dict(zip(X_test["user_id"],X_test["ts_listen"]))
	# with open("../files/user_test_listen_time_dict.json","w") as outfile:
	# 	json.dump(user_test_listen_dict,outfile)
	with open("../files/user_test_listen_time_dict.json") as datafile:
		user_test_listen_dict = json.load(datafile)
	def time_to_test_listen_func(row):
		user,ts_listen = row["user_id"],row["ts_listen"]
		date_of_listen = datetime.fromtimestamp(ts_listen).date()
		test_listen_time = user_test_listen_dict[unicode(str(int(user)),"utf-8")]
		test_listen_time = datetime.fromtimestamp(test_listen_time).date()
		return (test_listen_time-date_of_listen).days
	X_train["time_to_test_listen"] = X_train.apply(time_to_test_listen_func,axis=1)
	X_test["time_to_test_listen"] = X_test.apply(time_to_test_listen_func,axis=1)

def number_of_rows_for_user(X_train,X_test):
	# X_train = pd.read_csv("../data/train.csv")
	# user_pivot = pd.DataFrame(pd.pivot_table(X_train,index="user_id",values="is_listened",aggfunc=len))
	# user_len_dict = dict(zip(user_pivot.index,user_pivot.is_listened))
	# with open("../files/user_len_dict.json","w") as outfile:
	# 	json.dump(user_len_dict,outfile)
	with open("../files/user_len_dict.json") as datafile:
		user_len_dict = json.load(datafile)
	X_train["user_number_of_rows"] = X_train["user_id"].apply(lambda x: user_len_dict[str(x)])
	X_test["user_number_of_rows"] = X_test["user_id"].apply(lambda x: user_len_dict[str(x)])

def first_time_on_flow(X_train,X_test):
	# subset = X_train[X_train.listen_type==1]
	# min_time_stamp_pivot = pd.DataFrame(pd.pivot_table(subset,index="user_id",values="ts_listen",aggfunc=np.min))
	# user_list = min_time_stamp_pivot.index.values
	# ts_list = min_time_stamp_pivot["ts_listen"].ravel()
	# dictionary = dict()
	# for i in range(0,len(user_list)):
	# 	dictionary[user_list[i]] = ts_list[i]
	# # dictionary = dict(zip(min_time_stamp_pivot.index,min_time_stamp_pivot.ts_listen))
	# with open("../files/user_min_flow_timestamp.json",'w') as outfile:
	# 	json.dump(dictionary,outfile)
	def is_first_instance_check(row):
		user,timestamp = row["user_id"],row["ts_listen"]
		if user in user_list:
			if timestamp == new_dict[user]:
				return 1
			else:
				return 0
		elif row["listen_type"] == 1:
			return 1
		else:
			return 0
	with open("../files/user_min_flow_timestamp.json") as datafile:
		user_min_timestamp_dict = json.load(datafile)
	user_list = user_min_timestamp_dict.keys()
	new_dict = dict()
	for user in user_list:
		new_dict[int(user)] = user_min_timestamp_dict[user]
	user_list = set(new_dict.keys())
	X_train["is_first_flow_instance"] = X_train.apply(is_first_instance_check,axis=1)
	X_test["is_first_flow_instance"] = X_test.apply(is_first_instance_check,axis=1)

def avg_time_user_listen_song(X_train,X_test):	
	# subset = X_train[X_train.is_listened==1]
	# pivot = pd.DataFrame(pd.pivot_table(subset,values="user_id",index="media_id",aggfunc=[len,lambda x: len(x.unique())]))
	# pivot["avg_times_user_listens"] = pivot["len"]/pivot["<lambda>"]
	# dictionary = dict(zip(pivot.index,pivot.avg_times_user_listens))
	# train_songs = X_train["media_id"].unique()
	# test_songs = X_test["media_id"].unique()
	# songs_subset = subset["media_id"].unique()
	# for song in train_songs:
	# 	if song not in songs_subset:
	# 		dictionary[song] = 0
	# test_songs_not_in_train = []
	# for song in test_songs:
	# 	if song not in train_songs:
	# 		dictionary[song] = 1
	# with open("../files/song_avg_times_user_listens_crossval.json",'w') as outfile:
	# 	json.dump(dictionary,outfile)
	with open("../files/song_avg_times_user_listens_crossval.json") as datafile:
		dictionary = json.load(datafile)
	X_train["song_avg_times_listened"] = X_train["media_id"].apply(lambda x: dictionary[str(x)])
	X_test["song_avg_times_listened"] = X_test["media_id"].apply(lambda x: dictionary[str(x)])

def platform_family_analysis():
	X_train = pd.read_csv("../data/train.csv")
	X_test = pd.read_csv("../data/test.csv")
	print pd.pivot_table(X_train,values="is_listened",index="platform_family",aggfunc=[len,np.mean,np.median])


def song_age(X_train,X_test):
	X_train["song_age"] = X_train["release_date"].apply(lambda x: 2017 - int(x/10000))
	X_test["song_age"] = X_test["release_date"].apply(lambda x: 2017 - int(x/10000))

def user_age_song_age(X_train,X_test):
	X_train["song_age"] = X_train["release_date"].apply(lambda x: 2017 - int(x/10000))
	X_test["song_age"] = X_test["release_date"].apply(lambda x: 2017 - int(x/10000))
	def mult_user_song_age(row):
		return row["user_age"]*row["song_age"]
	X_train["user_song_age_mult"] = X_train.apply(mult_user_song_age,axis=1)
	X_test["user_song_age_mult"] = X_test.apply(mult_user_song_age,axis=1)

def number_of_songs_before():
	# total_rows = 7558834
	X_train = pd.read_csv("../data/train.csv",nrows=10000)
	subset = X_train[["user_id","media_id","ts_listen"]]
	def create_key(row):
		key = str(row["user_id"])+str(row["ts_listen"])
		return key
	subset["user_tslisten_key"] = subset.apply(create_key,axis=1)	
	# unique_user_list = subset["user_id"].unique()
	# for user in unique_user_list:
	# 	user_data = subset[subset.user_id==user]
	# 	user_data["media_id"].unique()

	# def check_in_subset(row):
	# 	user = row["user_id"]
	# 	ts_listen = row["ts_listen"]
	# 	data_subset = subset[(subset.user_id==user)&(subset.ts_listen<ts_listen)]
	# 	return len(data_subset)
	# X_train["number_of_songs_before"] = X_train.apply(check_in_subset,axis=1)
	# X_test["number_of_songs_before"] = X_test.apply(check_in_subset,axis=1)

def add_genre_is_listened_avg(X_train,X_test,validation):
	# unique_genre_list = X_train["genre_id"].unique().ravel()
	# genre_pivot = pd.pivot_table(X_train,values="is_listened",index="genre_id",aggfunc=np.mean)
	# genre_list = genre_pivot.index.values
	# avg_list_list = list(genre_pivot)
	# test_genre_list = X_test["genre_id"].unique().ravel()
	# test_genre_not_in_train = []
	# for test_genre in test_genre_list:
	# 	if test_genre not in genre_list:
	# 		test_genre_not_in_train.append(test_genre)
	# genre_avg_list_dict = dict()
	# test_genre_list = X_test["genre_id"].unique().ravel()
	# for i in range(0,len(genre_list)):
	# 	genre_avg_list_dict[genre_list[i]] = avg_list_list[i]
	# for test_genre in test_genre_not_in_train:
	# 	genre_avg_list_dict[test_genre] = 0.68
	# with open("../files/genre_avg_list_dict.json",'w') as outfile:
	# 	json.dump(genre_avg_list_dict,outfile)
	if validation==1:
		with open("../files/genre_avg_list_crossval_dict.json") as datafile:
			genre_avg_list_dict = json.load(datafile)
	else:
		with open("../files/genre_avg_list_dict.json") as datafile:
			genre_avg_list_dict = json.load(datafile)
	X_train["genre_list_avg"] = X_train["genre_id"].apply(lambda x: genre_avg_list_dict[str(x)])
	X_test["genre_list_avg"] = X_test["genre_id"].apply(lambda x: genre_avg_list_dict[str(x)])

def add_song_is_listened_avg(X_train,X_test,validation):
	#subset = X_train[["media_id","is_listened"]]
	#song_pivot = pd.DataFrame(pd.pivot_table(subset,values="is_listened",index="media_id",aggfunc=np.mean))
	#song_avg_list_dict = dict(zip(song_pivot.index.values,song_pivot["is_listened"]))
	#val_song_list = X_test["media_id"].unique().ravel()
	#X_test_act = pd.read_csv("../data/test.csv")
	#test_song_list = list(X_test_act["media_id"].unique().ravel())
	#existing_song_list = set(song_avg_list_dict.keys())
	#for song in val_song_list:
	#	if song not in test_song_list:
	#		test_song_list.append(song)
	#for song in test_song_list:
	#	if song not in existing_song_list:
	#		song_avg_list_dict[song] = 0.68
	#with open("../files/song_avg_list_crossval_dict.json",'w') as outfile:
	#	json.dump(song_avg_list_dict,outfile)
	if validation==1:
		with open("../files/song_avg_list_crossval_dict.json") as datafile:
			song_avg_list_dict = json.load(datafile)
	else:
		with open("../files/song_avg_list_dict.json") as datafile:
			song_avg_list_dict = json.load(datafile)
	X_train["song_list_avg_new"] = X_train["media_id"].apply(lambda x: song_avg_list_dict[str(x)])
	X_test["song_list_avg_new"] = X_test["media_id"].apply(lambda x: song_avg_list_dict[str(x)])
	X_train["song_list_avg"] = X_train["song_list_avg_new"]
	X_test["song_list_avg"] = X_train["song_list_avg_new"]
	del X_train["song_list_avg_new"]
	del X_test["song_list_avg_new"]



def add_is_listened_avg(X_train,X_test,validation):
	#subset = X_train[["user_id","is_listened"]]
	#user_pivot = pd.DataFrame(pd.pivot_table(subset,values="is_listened",index="user_id",aggfunc=np.mean))
	#user_avg_list_dict = dict(zip(user_pivot.index.values,user_pivot["is_listened"]))
	#with open("../files/user_avg_dict_v5_val_set.json","w") as outfile:
	#	json.dump(user_avg_list_dict,outfile)
	#with open("../files/user_total_len_crossval_dict.json") as datafile:
        #       user_total_len_dict = json.load(datafile)if validation == 0:
	with open("../files/user_avg_dict_v5_val_set.json") as datafile:
		user_avg_dict = json.load(datafile)
	#with open("../files/user_total_len_crossval_dict.json") as datafile:
	#	user_total_len_dict = json.load(datafile)
        
	user_dict_new = dict()
	for key in user_avg_dict.keys():
    		user_dict_new[int(key)] = user_avg_dict[key]
	#def standard_deviation_cal(row):
	#	user_id = row["user_id"]
	#	avg = row["is_listened_avg"]
	#	total_len = float(user_total_len_dict[str(int(user_id))])
	#	return (avg*(1-avg))/total_len
	user_set = set(user_dict_new.keys())
	X_train["is_listened_avg_new"] = X_train["user_id"].apply(lambda x: user_dict_new[int(x)] if x in user_set else 0.68)
	X_test["is_listened_avg_new"] = X_test["user_id"].apply(lambda x: user_dict_new[int(x)] if x in user_set else 0.68)
	# X_train["is_listened_std"] = X_train.apply(standard_deviation_cal,axis=1) 
	# X_test["is_listened_std"] = X_test.apply(standard_deviation_cal,axis=1)
	X_train["is_listened_avg"] = X_train["is_listened_avg_new"]
	X_test["is_listened_avg"]= X_test["is_listened_avg_new"]
	del X_train["is_listened_avg_new"]
	del X_test["is_listened_avg_new"]

def add_is_listened_on_flow_avg(X_train,X_test):
	with open("../files/user_both_type_listen_avg_crossval.json") as datafile:
		user_both_type_avg_dict = json.load(datafile)
	X_train["is_listened_on_0_avg"] = X_train["user_id"].apply(lambda x: float(user_both_type_avg_dict[str(x)][0]))
	X_test["is_listened_on_0_avg"] = X_test["user_id"].apply(lambda x: float(user_both_type_avg_dict[str(x)][0]))
	X_train["is_listened_on_1_avg"] = X_train["user_id"].apply(lambda x: float(user_both_type_avg_dict[str(x)][1]))
	X_test["is_listened_on_1_avg"] = X_test["user_id"].apply(lambda x: float(user_both_type_avg_dict[str(x)][1]))
	# full_train_data = pd.read_csv("../data/train.csv")
	# unique_user_list = full_train_data["user_id"].unique()
	# print "making pivots"
	# user_pivot0 = pd.pivot_table(X_train[X_train.listen_type==0],values="is_listened",index="user_id",aggfunc=np.mean)
	# user_0_list_dict = dict()
	# user_0_list = user_pivot0.index.values
	# avg_list0 = list(user_pivot0)
	# user_pivot1 = pd.pivot_table(X_train[X_train.listen_type==1],values="is_listened",index="user_id",aggfunc=np.mean)
	# user_1_list = user_pivot1.index.values
	# avg_list1 = list(user_pivot1)
	# user_1_list_dict = dict()
	# print "creating individual dictionaries"
	# for i in range(0,len(user_0_list)):
	# user_1_list = user_pivot1.index.values
	# avg_list1 = list(user_pivot1)
	# user_1_list_dict = dict()
	# print "creating individual dictionaries"
	# for i in range(0,len(user_0_list)):
	# 	user_0_list_dict[user_0_list[i]] = avg_list0[i]
	# for j in range(0,len(user_1_list)):
	# 	user_1_list_dict[user_1_list[j]] = avg_list1[j]
	# user_both_type_avg_dict = dict()
	# for k in range(0,len(unique_user_list)):
	# 	if k%1000==0:
	# 		print k
	# 	user = unique_user_list[k]
	# 	try:
	# 		listen_type_0_avg = user_0_list_dict[user]
	# 	except:
	# 		listen_type_0_avg = 0

	# 	try:
	# 		listen_type_1_avg = user_1_list_dict[user]
	# 	except:
	# 		listen_type_1_avg = 0
	# 	user_both_type_avg_dict[user] = [listen_type_0_avg,listen_type_1_avg]

	# with open("../files/user_both_type_listen_avg_crossval.json","w") as outfile:
	# 	json.dump(user_both_type_avg_dict,outfile)
	
def add_user_total_len(X_train,X_test):
	# subset = X_train[["user_id","is_listened"]]
	# user_pivot = pd.pivot_table(subset,values="is_listened",index="user_id",aggfunc=np.sum)
	# user_total_len_dict = dict()
	# user_list = user_pivot.index.values
	# len_list = list(user_pivot)
	# for i in range(0,len(len_list)):
	# 	user_total_len_dict[user_list[i]] = len_list[i]
	# with open("../files/user_total_len_crossval_dict.json","w") as outfile:
	# 	json.dump(user_total_len_dict,outfile)
	with open("../files/user_total_len_crossval_dict.json") as datafile:
		user_total_len_dict = json.load(datafile)
	X_train["user_listened_len"] = X_train["user_id"].apply(lambda x: user_total_len_dict[str(x)])
	X_test["user_listened_len"] = X_test["user_id"].apply(lambda x: user_total_len_dict[str(x)])

def one_hot_encoding_context(X_train,X_test):
	print X_train.context_type.value_counts()
	pivot_df = pd.DataFrame(pd.pivot_table(X_train,values="is_listened",index="context_type",aggfunc=[len,np.mean]))
