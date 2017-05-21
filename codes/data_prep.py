import pandas as pd

def import_validation_dataset():
    X_train = pd.read_csv("../data/trainval_train_v3.csv")
    X_test = pd.read_csv("../data/trainval_val_v3.csv")
    # X_train = X_train[X_train.listen_type==0]
    # test_song_list = X_test["media_id"].unique()
    # X_train = X_train[X_train.media_id.isin(test_song_list)]
    return [X_train,X_test]

def import_train_test_set():
	X_train = pd.read_csv("../data/train.csv")
	X_test = pd.read_csv("../data/test.csv")
	X_train = X_train[X_train.listen_type==0]
	test_song_list = X_test["media_id"].unique()
	X_train = X_train[X_train.media_id.isin(test_song_list)]
	return [X_train,X_test]