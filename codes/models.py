import xgboost as xgb
def xgboost_model(X_train,y):
	param = {}
	param['objective'] = 'multi:softprob'
	param['eta'] = 0.02
	param['max_depth'] = 35
	param['silent'] = 1
	param['num_class'] = 2
	param['eval_metric'] = "auc"
	param['min_child_weight'] = 1
	param['subsample'] = 0.7
	param['colsample_bytree'] = 0.7
	param['seed'] = 321
	param['nthread'] = 8
	num_rounds = 10

	xgtrain = xgb.DMatrix(X_train,y)
	clf = xgb.train(param, xgtrain, num_rounds)
	return clf
