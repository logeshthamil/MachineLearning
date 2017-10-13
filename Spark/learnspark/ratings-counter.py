from pyspark import SparkConf, SparkContext
import collections
import os

os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
conf = SparkConf().setMaster("local").setAppName("RatingsHistogram")
sc = SparkContext(conf=conf)

lines = sc.textFile("/home/lt/quanox/QX_Data/test_recommender_datasets/movielens/ml-1m/ratings.dat")
ratings = lines.map(lambda x: x.split(":")[4])
result = ratings.countByValue()

sortedResults = collections.OrderedDict(sorted(result.items()))
for key, value in sortedResults.items():
    print("%s %i" % (key, value))
