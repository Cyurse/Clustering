import time
import math
import xgboost as xgb
import pandas as pd
import json
import pyspark
import sys


def weight(active_item, other_item):
    pearson_corr = 0
    active_user = item_user[active_item].keys()
    other_user = item_user[other_item].keys()

    co_users = set(active_user) & set(other_user)

    num = 0
    denominator1 = 0
    denominator2 = 0
    for i in co_users:
        active_rating = item_user[active_item][i] - item_avg[active_item][0]
        other_rating = item_user[other_item][i] - item_avg[other_item][0]
        num = num + active_rating * other_rating
        denominator1 = denominator1 + active_rating * active_rating
        denominator2 = denominator2 + other_rating * other_rating

    if num > 0:
        pearson_corr = num / math.sqrt(denominator1 * denominator2)
    weight_m[(active_item, other_item)] = pearson_corr
    return pearson_corr


def predict(test_pair):
    user_id = test_pair[0]
    business_id = test_pair[1]
    sim = 0
    businesses = user_avg[user_id][0]
    pearson_corr = []
    for i in businesses:
        if user_id not in user_avg.keys() and business_id not in item_avg.keys():
            return (test_pair[0],test_pair[1]),2.5
        elif business_id not in item_avg.keys():
            return (test_pair[0],test_pair[1]),user_avg[user_id][1]
        elif user_id not in user_avg.keys() or item_avg[business_id][2]:
            return (test_pair[0], test_pair[1]), item_avg[business_id][0]
        else:
            if user_id in item_user[business_id].keys():
                return (test_pair[0], test_pair[1]), item_user[business_id][user_id]
            else:
                if i != business_id:
                    if (business_id, i) in weight_m:
                        sim = weight_m[(business_id, i)]
                    elif (i, business_id) in weight_m:
                        sim = weight_m[(i, business_id)]
                    elif i in item_user.keys() and business_id in item_user.keys():
                        sim = weight(business_id, i)
                    else:
                        sim = 0
                if sim > 0:
                    pearson_corr.append((i, sim))

    num = 0
    den = 0
    if len(pearson_corr) == 0 or len(pearson_corr) == 1:
        return (test_pair[0], test_pair[1]),user_avg[user_id][1]
    for i in pearson_corr:
        rating = item_user[i[0]][user_id]
        num = num + rating * i[1]
        den = den + abs(i[1])

    if den == 0:
        return (test_pair[0],test_pair[1]),0
    else:
        return (test_pair[0],test_pair[1]),num/den


start_time = time.time()

sc = pyspark.SparkContext('local[*]', 'task2_3')
sc.setLogLevel("ERROR")

input_file1 = sys.argv[1] + 'yelp_train.csv'
input_file2 = sys.argv[1] + 'business.json'
input_file3 = sys.argv[1] + 'user.json'
test_file = sys.argv[2]
out_file = sys.argv[3]


read_data1 = sc.textFile(input_file1)
read_data2 = sc.textFile(input_file2)
read_data3 = sc.textFile(input_file3)
read_data_test = sc.textFile(test_file)
rdd1 = read_data1.filter(lambda a: a != 'user_id,business_id,stars').map(lambda a: a.split(','))
rdd3 = read_data3.map(json.loads).map(lambda idx: (idx['user_id'], idx['average_stars']))
test_rdd = read_data_test.filter(lambda a: a != 'user_id,business_id,stars').map(lambda a: a.split(','))

# Item_based
# (item, (average, dis_to_origin, count>20))
item_avg = rdd1.map(lambda a: (a[1], (float(a[2]), float(a[2])**2, 1))).\
    reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])).\
    map(lambda a: (a[0], (a[1][0]/a[1][2], pow(a[1][1], 0.5), a[1][2] > 20))).collectAsMap()
# (user, (item_list, average))
user_avg = rdd1.map(lambda a: (a[0], ([a[1]], float(a[2]), 1))).\
    reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])).map(lambda a: (a[0], (a[1][0], a[1][1]/a[1][2]))).\
    collectAsMap()

item_user = rdd1.map(lambda a: (a[1], {a[0]: float(a[2])})).reduceByKey(lambda a, b: {**a, **b}).collectAsMap()
weight_m = {}
# prediction
test_pairs = test_rdd.map(lambda x: (x[0], x[1]))
prediction = test_pairs.map(lambda a: predict(a))

# model_based
user_rdd = rdd1.map(lambda a: (a[0], 1)).reduceByKey(lambda a, b: a + b).join(rdd3).collectAsMap()
business_rdd = read_data2.map(json.loads).map(lambda idx: (idx['business_id'], (idx['stars'], idx['review_count'])))\
    .collectAsMap()

train_data = rdd1.\
    map(lambda a: (a[0], a[1], user_rdd[a[0]][1], user_rdd[a[0]][0], float(business_rdd[a[1]][0]),
                   float(business_rdd[a[1]][1]), float(a[2]))).collect()
test_data = prediction. \
    map(lambda a: (a[0][0], a[0][1], user_rdd[a[0][0]][1], user_rdd[a[0][0]][0], float(business_rdd[a[0][1]][0]),
                   float(business_rdd[a[0][1]][1]), a[1])).collect()

train_df = pd.DataFrame(train_data, columns=['user_id', 'business_id', 'user_average', 'n_user_review',
                                             'business_average', 'n_business_review', 'stars'])
test_df = pd.DataFrame(test_data, columns=['user_id', 'business_id', 'user_average', 'n_user_review',
                                           'business_average', 'n_business_review', 'item_based_preds'])

Y_train = train_df.stars.values
X_train = train_df.drop(["stars"], axis=1)
X_train = X_train.drop(['user_id'], axis=1)
X_train = X_train.drop(['business_id'], axis=1)
X_train = X_train.values

X_test = test_df.drop(['user_id'], axis=1)
X_test = X_test.drop(['business_id'], axis=1)
X_test = X_test.drop(['item_based_preds'], axis=1)
X_test = X_test.values

model = xgb.XGBRegressor()
model.fit(X_train, Y_train)
model_based_preds = model.predict(data=X_test)
item_based_preds = test_df.item_based_preds.values

final_preds = []
for i in range(len(model_based_preds)):
    if item_based_preds[i] == 0:
        final_preds.append(model_based_preds[i])
    elif abs(model_based_preds[i] - item_based_preds[i]) < 0.6:

        final_preds.append(0.15 * item_based_preds[i] + 0.85 * model_based_preds[i])

    else:
        final_preds.append(model_based_preds[i])
        
output = pd.DataFrame()
output["user_id"] = test_df.user_id.values
output["business_id"] = test_df.business_id.values
output["prediction"] = final_preds
output.to_csv(out_file, index=False)

# print(time.time()-start_time)