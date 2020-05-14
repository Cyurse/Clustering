import xgboost as xgb
import pandas as pd
import json
import pyspark
import time
import sys

start_time = time.time()

sc = pyspark.SparkContext('local[*]', 'task2_2')
sc.setLogLevel("ERROR")

input_file1 = sys.argv[1]+'yelp_train.csv'
input_file2 = sys.argv[1]+'review_train.json'
input_file3 = sys.argv[1]+'business.json'
test_file = sys.argv[2]
out_file = sys.argv[3]


read_data1 = sc.textFile(input_file1)
read_data2 = sc.textFile(input_file2)
read_data3 = sc.textFile(input_file3)
read_data_test = sc.textFile(test_file)
rdd1 = read_data1.filter(lambda a: a != 'user_id,business_id,stars').map(lambda a: a.split(','))
rdd2 = read_data2.map(json.loads).map(lambda idx: (idx['user_id'], 1)).reduceByKey(lambda a, b: a+b)
test_rdd = read_data_test.filter(lambda a: a != 'user_id,business_id,stars').map(lambda a: a.split(','))
user_avg = rdd1.map(lambda a: (a[0], (float(a[2]), 1))).reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1])).\
    map(lambda a: (a[0], a[1][0]/a[1][1]))

user_rdd = user_avg.join(rdd2).collectAsMap()
business_rdd = read_data3.map(json.loads).map(lambda idx: (idx['business_id'], (idx['stars'], idx['review_count'])))\
    .collectAsMap()

train_data = rdd1.\
    map(lambda a: (a[0], a[1], user_rdd[a[0]][0], user_rdd[a[0]][1], business_rdd[a[1]][0],
                   business_rdd[a[1]][1], a[2])).collect()
test_data = test_rdd. \
    map(lambda a: (a[0], a[1], user_rdd[a[0]][0], user_rdd[a[0]][1], business_rdd[a[1]][0],
                   business_rdd[a[1]][1])).collect()

train_df = pd.DataFrame(train_data, columns=['user_id', 'business_id', 'user_average', 'n_user_review',
                                             'business_average', 'n_business_review', 'stars'])
test_df = pd.DataFrame(test_data, columns=['user_id', 'business_id', 'user_average', 'n_user_review',
                                           'business_average', 'n_business_review'])

Y_train = train_df.stars.values
X_train = train_df.drop(["stars"], axis=1)
X_train = X_train.drop(['user_id'], axis=1)
X_train = X_train.drop(['business_id'], axis=1)
X_train = X_train.values

X_test = test_df.drop(['user_id'], axis=1)
X_test = X_test.drop(['business_id'], axis=1)
X_test = X_test.values

model = xgb.XGBRegressor()
model.fit(X_train, Y_train)
preds = model.predict(data=X_test)

output = pd.DataFrame()
output["user_id"] = test_df.user_id.values
output["business_id"] = test_df.business_id.values
output["prediction"] = preds
output.to_csv(out_file, index=False)
# print(time.time()-start_time)
