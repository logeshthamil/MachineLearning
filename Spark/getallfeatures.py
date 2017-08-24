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
        a = int(a)
        if (a < 10) or (a > 80):
            a = None
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
        weights1 = []
        weights2 = []
        timedata = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                    15: 0,
                    16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
        weekday = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        devices = {"Macintosh": 0,
                   "iPhone": 0,
                   "Windows NT 10.0": 0,
                   "Linux": 0,
                   "Windows NT 6.1": 0,
                   "iPad": 0,
                   "Windows NT 6.1) AppleWebKit/537.36 ": 0,
                   "Windows NT 6.3": 0,
                   "Windows NT 5.1": 0,
                   "compatible": 0,
                   "Windows NT 6.0": 0,
                   "Windows": 0,
                   "X11": 0,
                   "Windows NT 6.0) AppleWebKit/537.36 ": 0,
                   "Windows NT 6.2": 0,
                   "LTLM888 ": 0,
                   "Windows NT 10.0) AppleWebKit/537.36 ": 0,
                   "Mobile": 0,
                   "iPod touch": 0,
                   "Android 6.0.1": 0,
                   "Windows NT 5.1) AppleWebKit/537.36 ": 0,
                   "Android 7.0": 0,
                   "Windows Phone 10.0": 0,
                   "Windows NT 6.2) AppleWebKit/537.22 ": 0,
                   "Android 7.1.1": 0,
                   "Windows NT 10.0.14393.1198": 0}

        ## fill weights
        for clusternum in range(clusterscount):
            cluster = clusters[clusternum]
            weights1.append(float(cluster.get("weight")))
        weights1 = [float(i) / sum(weights1) for i in weights1]

        # weights2.append(float(document.get("totalVisits")))
        weights2.append(float(document.get("rtlluVisits")))
        weights2.append(float(document.get("editusVisits")))
        weights2.append(float(document.get("habiterVisits")))
        weights2 = [float(i) / sum(weights2) for i in weights2]

        ## fill timestamp
        for visits in document.get("lastVisits"):
            timedata[int(visits.get("timestamp")[11:13])] += 1
            dt = datetime.datetime.strptime(visits.get("timestamp")[0:10], "%Y-%m-%d")
            weekday[int(dt.weekday())] += 1
        weekday = weekday.values()
        timedata = timedata.values()
        weekday = [float(i) / sum(weekday) for i in weekday]
        timedata = [float(i) / sum(timedata) for i in timedata]

        ## browser info
        for visits in document.get("lastVisits"):
            b = visits.get("browser").split('(')[1].split(';')[0]
            devices[b] += 1
        devices = devices.values()
        devices = [float(i) / sum(devices) for i in devices]

        if age is not None:
            towrite = str(age) + ',' + ','.join(map(str, devices)) + ',' + ','.join(map(str, timedata)) + '\n'
            featuresfile.write(towrite)
    featuresfile.close()


features_file = "/tmp/rtleditus/agepredictionfeatures.dat"
extract_cluster_tofile(features_file=features_file)
featuresmat = numpy.loadtxt(fname=features_file, delimiter=',', dtype=str)
featuresmat[featuresmat == 'nan'] = '0.0'
age = featuresmat[:, 0]
feature = featuresmat[:, 1:]
featurem = feature[~(feature == '0.0')[:].all(1)]
age = age[~(feature == '0.0')[:].all(1)]
featurem = featurem.astype(float)
age = age.astype(float)

from sklearn import svm, ensemble
from sklearn.model_selection import KFold
random_forest = ensemble.RandomForestRegressor(n_estimators=100)
gradient_boost = ensemble.GradientBoostingRegressor(n_estimators=100)
svm_ = svm.SVR(kernel='linear')
svm_poly = svm.SVR(kernel='poly', degree=3)
svm_rbf = svm.SVR(kernel='rbf', degree=10)
n_folds = 5

regres = random_forest
kf = KFold(n_splits=n_folds)
original = []
predicted = []
for train_index, test_index in kf.split(featurem, age):
    X_train, X_test = featurem[train_index], featurem[test_index]
    y_train, y_test = age[train_index], age[test_index]
    regres.fit(X_train, y_train)
    original = numpy.concatenate((original, y_test))
    predicted = numpy.concatenate((predicted, regres.predict(X_test)))

for p,o in zip(predicted, original):
    print(p, ' -> ', o)
## Plot outputs
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(original, predicted, edgecolors=(0, 0, 0))
ax.plot([10, 80], [10, 80], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

from sklearn.metrics import mean_absolute_error
print("Mean Squared Error", mean_absolute_error(original, predicted))

# # spark machine learning
# from pyspark import SparkConf, SparkContext
# from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
# from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
#
# # set up spark
# conf = SparkConf().setMaster("local").setAppName("agePrediction")
# sc = SparkContext(conf=conf)
#
# # Load and parse the data
# def parsePoint(line):
#     values = [float(x) for x in line.split(',')]
#     return LabeledPoint(values[0], values[1:])
#
# data = sc.textFile(features_file)
# parsedData = data.map(parsePoint)
#
# # Build the model
# # model = LinearRegressionWithSGD.train(parsedData, iterations=100, step=0.00000001)
# model = GradientBoostedTrees.trainRegressor(parsedData, numIterations=100, categoricalFeaturesInfo={})
#
# # Evaluate the model on training data
# valuesAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
# MSE = valuesAndPreds \
#     .map(lambda vp: (vp[0] - vp[1])**2) \
#     .reduce(lambda x, y: x + y) / valuesAndPreds.count()
# print("Mean Squared Error = " + str(MSE))

