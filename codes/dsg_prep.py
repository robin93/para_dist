import json

# with open("/Users/ROBIN/Desktop/dsg17/prep/kaggle_comps/2sigma_rental_listing/train.json") as datafile:
# 	data_object = json.load(datafile)

with open("/Users/ROBIN/Desktop/dsg17/prep/kaggle_comps/2sigma_rental_listing/test.json") as datafile:
	data_object = json.load(datafile)

print type(data_object)
print len(data_object.keys())

for key in data_object.keys():
	print key, type(data_object[key]),len(data_object[key])

# for key in data_object["listing_id"].keys():
# 	print key,data_object["listing_id"][key]