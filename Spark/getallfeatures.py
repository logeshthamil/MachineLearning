# coding=utf-8
import pymongo
import os
import io
import numpy
import datetime, time

uri = "mongodb://" + "qxadmin" + ":" + "qx16admin" + "@" + "127.0.0.1" + "/" + "?authSource=" + "users"
client = pymongo.MongoClient(uri)
users_collection = client.rtleditus.Users


def getagefromyear(year=None):
    """
    Assign a class label for the year label
    :param year: the year in which the user is born
    :return: the class label
    """
    if year is None:
        print("Please enter the year to assign class to them")
    try:
        t = datetime.datetime.today()
        b = datetime.datetime.strptime(str(year), '%Y')
        a = (t - b).days / 365
    except:
        a = None
    return a

def plot_age(agearray=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde, kde
    density = kde.gaussian_kde(agearray)
    x = numpy.arange(0., 100, .1)
    plt.plot(x, density(x))
    plt.show()

def extract_cluster_tofile(features_file=None):
    if not os.path.exists(os.path.dirname(features_file)):
        os.mkdir(os.path.dirname(features_file))
    featuresfile = io.open(file=features_file, mode='w')
    users_with_age = users_collection.find({"birthYear": {"$ne": ""}})
    for document in users_with_age:
        year = document.get("birthYear")
        age = getagefromyear(year)
        clusters = document.get("clusters")
        clusterscount = len(clusters)
        weights = []
        contentid = []
        browser = []
        time = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
        weekday = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        ## fill weights
        for clusternum in range(clusterscount):
            cluster = clusters[clusternum]
            weights.append(str(cluster.get("weight")))
        weights.append(str(document.get("totalVisits")))
        weights.append(str(document.get("rtlluVisits")))
        weights.append(str(document.get("editusVisits")))
        weights.append(str(document.get("habiterVisits")))

        ## fill timestamp
        for visits in document.get("lastVisits"):
            time[int(visits.get("timestamp")[11:13])] += 1
            dt = datetime.datetime.strptime(visits.get("timestamp")[0:10], "%Y-%m-%d")
            weekday[int(dt.weekday())] += 1
        if age is not None:
            towrite = str(age) + ',' + ','.join(weights) + ','.join(contentid) + ','.join(browser) + ','.join(
                list(map(str, time.values()))) + ','.join(list(map(str, weekday.values()))) + '\n'
            featuresfile.write(towrite)
    featuresfile.close()


features_file = "/tmp/rtleditus/agepredictionfeatures.dat"
# extract_cluster_tofile(features_file=features_file)
featuresmat = numpy.loadtxt(fname=features_file, delimiter=',', dtype=str)
featuresmat[featuresmat == 'nan'] = '0.0'
age = featuresmat[:, 0]
feature = featuresmat[:, 1:]
featurem = feature[~(feature == '0.0')[:].all(1)]
gender = age[~(feature == '0.0')[:].all(1)]
featurem = featurem.astype(float)
age = age.astype(float)


# spark machine learning
from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

# set up spark
conf = SparkConf().setMaster("local").setAppName("agePrediction")
sc = SparkContext(conf=conf)

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

data = sc.textFile(features_file)
parsedData = data.map(parsePoint)

# Build the model
# model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)
model = GradientBoostedTrees.trainRegressor(parsedData, numIterations=100, categoricalFeaturesInfo={})

# Evaluate the model on training data
valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
MSE = valuesAndPreds \
    .map(lambda vp: (vp[0] - vp[1])**2) \
    .reduce(lambda x, y: x + y) / valuesAndPreds.count()
print("Mean Squared Error = " + str(MSE))

