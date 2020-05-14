import pyspark
import sys
from itertools import combinations
import time

start_time = time.time()

def min_hashing(c):
    min_value = [min((a*x + 1) % m for x in c[1]) for a in a_list]
    return (c[0], min_value)


def LSH_hash(s):
    bucket = []
    b = int(len(s[1])/r)
    for band in range(0, b):
        bucket.append(((band, tuple(s[1][band*r:(band+1)*r])), [s[0]]))
    return bucket

def generate_candidate(s):
    bucket = s[1]
    bucket.sort()
    candidates = [x for x in combinations(bucket, 2)]
    return candidates


def jaccard_similarity(c):
    users1 = set(similarity_rdd[c[0]])
    users2 = set(similarity_rdd[c[1]])
    similarity = len(users1.intersection(users2))/len(users1.union(users2))
    return (c, similarity)


input_file_path = sys.argv[1]
output_file_path = sys.argv[2]

sc = pyspark.SparkContext('local[*]', 'task1')
sc.setLogLevel("ERROR")

read_data = sc.textFile(input_file_path)
rdd = read_data.filter(lambda a: a != 'user_id,business_id,stars').map(lambda a: a.split(',')).persist()
users = rdd.map(lambda a: a[0]).distinct().collect()
businesses = rdd.map(lambda a: a[1]).distinct().collect()
similarity_rdd = rdd.map(lambda a: (a[1], [a[0]])).reduceByKey(lambda a, b: a+b).collectAsMap()

a_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
          109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
m = len(users)
r = 2

i = 0
users_map = {}
for u in users:
    users_map[u] = i
    i = i+1

# min hashing
characteristic_matrix = rdd.map(lambda x: (x[1], [users_map[x[0]]])).reduceByKey(lambda a, b: a+b)
signature_matrix = characteristic_matrix.map(lambda c: min_hashing(c))

# LSH
sig_hash = signature_matrix.flatMap(lambda s: LSH_hash(s)).reduceByKey(lambda a, b: a+b)
candidates = sig_hash.flatMap(lambda s: generate_candidate(s)).distinct()

# Jaccard Similarity
similarities = candidates.map(lambda c: jaccard_similarity(c)).filter(lambda x: x[1] >= 0.5).\
    sortBy(lambda a: (a[0][0], a[0][1])).collect()

f = open(output_file_path, 'w')
f.write('business_id_1, business_id_2, similarity'+'\n')
for i in similarities:
    f.write(i[0][0] + ',' + i[0][1] + ',' + str(i[1]) + '\n')
f.close()
# print(time.time()-start_time)