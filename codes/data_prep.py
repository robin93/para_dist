import pandas as pd

def import_validation_dataset():
    X_train = pd.read_csv("../data/trainval_train_v6.csv")
    #X_test = pd.read_csv("../data/trainval_val_v6.csv")
    X_test = pd.read_csv("../data/test_v6.csv")
    # X_train = X_train[X_train.listen_type==0]
    X_test_act = pd.read_csv("../data/test.csv")
    test_song_list = X_test_act["media_id"].unique()
    X_train = X_train[X_train.media_id.isin(test_song_list)]
    return [X_train,X_test]

def import_train_test_set():
	X_train = pd.read_csv("../data/trainval_train_v6.csv")
	X_test = pd.read_csv("../data/test_v6.csv")
	#X_train = X_train[X_train.listen_type==0]
	test_song_list = X_test["media_id"].unique()
	X_train = X_train[X_train.media_id.isin(test_song_list)]
	return [X_train,X_test]
