import pyspark
import time
import math
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

input_file_path = sys.argv[1]
test_file_path = sys.argv[2]
output_file_path = sys.argv[3]

sc = pyspark.SparkContext('local[*]', 'task2_1')
sc.setLogLevel("ERROR")

read_data = sc.textFile(input_file_path)
rdd = read_data.filter(lambda a: a != 'user_id,business_id,stars').map(lambda a: a.split(',')).persist()
read_data2 = sc.textFile(test_file_path)
test_rdd = read_data2.filter(lambda a: a != 'user_id,business_id,stars').map(lambda a: a.split(',')).persist()

# (item, (average, dis_to_origin, count>20))
item_avg = rdd.map(lambda a: (a[1], (float(a[2]), float(a[2])**2, 1))).\
    reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])).\
    map(lambda a: (a[0], (a[1][0]/a[1][2], pow(a[1][1], 0.5), a[1][2] > 20))).collectAsMap()
# (user, (item_list, average))
user_avg = rdd.map(lambda a: (a[0], ([a[1]], float(a[2]), 1))).\
    reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1], a[2]+b[2])).map(lambda a: (a[0], (a[1][0], a[1][1]/a[1][2]))).\
    collectAsMap()

item_user = rdd.map(lambda a: (a[1], {a[0]: float(a[2])})).reduceByKey(lambda a, b: {**a, **b}).collectAsMap()
weight_m = {}
# prediction
test_pairs = test_rdd.map(lambda x: (x[0], x[1]))
prediction = test_pairs.map(lambda a: predict(a))

predictions = prediction.collect()
f = open(output_file_path, 'w')
f.write('user_id,business_id,prediction'+'\n')
for i in predictions:
    f.write(i[0][0] + ',' + i[0][1] + ',' + str(i[1]) + '\n')
f.close()
# print(time.time() - start_time)