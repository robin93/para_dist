import pandas as pd, numpy as np, sys,time, random, xgboost as xgb, models
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from datetime import datetime
import feature_engineering as fe, data_prep as dp
start_time = time.time()

def main():
    is_validation = int(raw_input("Enter 1 to run code of cross validation set or enter 0 otherwise: \n"))
    random.seed(321),np.random.seed(321)
    print "Reading data files"
    if is_validation==1:
        X_train,X_test = dp.import_validation_dataset()[0],dp.import_validation_dataset()[1]
    else:
        X_train,X_test = dp.import_train_test_set()[0],dp.import_train_test_set()[1]

    # print X_train.describe().transpose()
    print "Starting transformations"
    # X_train,X_test = transform_data(X_train),transform_data(X_test) 
    y = X_train['is_listened'].ravel()
    #fe.features_to_add(X_train,X_test,validation=is_validation)
    print "Normalizing high cordiality data"
    # normalize_high_cordiality_data()
    # transform_categorical_data()

    print "removing columns"
    remove_columns(X_train),remove_columns(X_test)

    print list(X_train),"\n","---",int((time.time() - start_time)/60), "minutes","Start fitting"
    y_train_true = X_train["is_listened"].ravel()
    del X_train["is_listened"]

    clf = models.xgboost_model(X_train,y)
    #clf = RandomForestClassifier(n_jobs=2,max_depth=13,n_estimators=20)
    #clf.fit(X_train,y)


    print "Fitted"
    if is_validation==1:
        print_roc_auc_score(clf,X_train,X_test,y_train_true)
    else:
        prepare_submission(clf,X_test)
    print "---",int((time.time() - start_time)/60), "minutes"


def print_roc_auc_score(model,X_train,X_test,y_train_true):
  y_true = X_test['is_listened'].ravel()
  del X_test["is_listened"]
  xgtest = xgb.DMatrix(X_test)
  preds = model.predict(xgtest)
  # preds = model.predict_proba(X_test)
  # preds_train = model.predict_proba(X_train)
  y_scores = preds[:,1]
  # y_scores_train = preds_train[:,1]
  print "Val AUC: ",roc_auc_score(y_true, y_scores),# "Train AUC: ", roc_auc_score(y_train_true,y_scores_train),

def prepare_submission(model,X_test):
    sub = pd.DataFrame(data = {'sample_id': X_test['sample_id'].ravel()})
    del X_test["sample_id"]
    # xgtest = xgb.DMatrix(X_test)
    # preds = model.predict(xgtest)
    preds = model.predict_proba(X_test)
    sub['is_listened'] = preds[:, 1]  
    sub.to_csv("../submissions/submission_10May.csv", index = False, header = True)

def transform_categorical_data():
    categorical = ['context_type', 'platform_name',
                   'platform_family','listen_type','user_gender']
                   
    for f in categorical:
        encoder = LabelEncoder()
        encoder.fit(list(X_train[f]) + list(X_test[f])) 
        X_train[f] = encoder.transform(X_train[f].ravel())
        X_test[f] = encoder.transform(X_test[f].ravel())



def remove_columns(X):
    columns = ['genre_id', 'album_id','media_id','context_type'
              ,'user_id','ts_listen',"release_date",'artist_id'
              ,'platform_name','platform_family',"media_duration","listen_type"]
    for c in columns:
        del X[c]

if __name__ == "__main__":
    main()

