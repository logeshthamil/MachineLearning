# coding=utf-8
import pymongo
import os
import io
import numpy
import datetime

uri = "mongodb://" + "qxadmin" + ":" + "qx16admin" + "@" + "127.0.0.1" + "/" + "?authSource=" + "users"
client = pymongo.MongoClient(uri)
users_collection = client.rtleditus.Users


def extract_cluster_tofile(features_file=None):
    if not os.path.exists(os.path.dirname(features_file)):
        os.mkdir(os.path.dirname(features_file))
    featuresfile = io.open(file=features_file, mode='w')
    users_with_gender = users_collection.find({"gender": {"$ne": ""}})
    for document in users_with_gender:
        gender = document.get("gender")
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

        # towrite = gender + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, weights2)) + ',' + ','.join(
        #     map(str, weekday)) + ',' + ','.join(map(str, timedata)) + '\n'
        towrite = gender + ',' + ','.join(map(str, devices)) + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, weights2)) + '\n'
        featuresfile.write(towrite)
    featuresfile.close()


features_file = "/tmp/rtleditus/genderpredictionfeatures2.dat"
# extract_cluster_tofile(features_file=features_file)
featuresmat = numpy.loadtxt(fname=features_file, delimiter=',', dtype=str)
featuresmat[featuresmat == 'nan'] = '0.0'
gender = featuresmat[:, 0]
gender[gender == 'm'] = 1
gender[gender == 'f'] = 0
feature = featuresmat[:, 1:]
featurem = feature[~(feature == '0.0')[:].all(1)]
gender = gender[~(feature == '0.0')[:].all(1)]
featurem = featurem.astype(float)
gender = gender.astype(float)

from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.svm import SVC, LinearSVC

n_folds = 5
clf = ensemble.GradientBoostingClassifier(n_estimators=100)
# clf = SVC()
# clf = LinearSVC()


kf = KFold(n_splits=n_folds)
original = []
predicted = []
for train_index, test_index in kf.split(featurem, gender):
    X_train, X_test = featurem[train_index], featurem[test_index]
    y_train, y_test = gender[train_index], gender[test_index]
    clf.fit(X_train, y_train)
    original = numpy.concatenate((original, y_test))
    predicted = numpy.concatenate((predicted, clf.predict(X_test)))

tn, fp, fn, tp = confusion_matrix(original, predicted).ravel()
print(confusion_matrix(original, predicted))
print("Accuracy: ", (tn+tp)/(tn+fp+fn+tp))


## pandas dataframe
# import pandas as pd
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as pyplot
# data = pd.read_csv(features_file, names=["gender", "Macintosh", "iPhone", "Windows NT 10.0", "Linux", "Windows NT 6.1", "iPad",
#                                          "Windows NT 6.1) AppleWebKit/537.36 ", "Windows NT 6.3", "Windows NT 5.1",
#                                          "compatible", "Windows NT 6.0", "Windows", "X11",
#                                          "Windows NT 6.0) AppleWebKit/537.36 ", "Windows NT 6.2", "LTLM888 ",
#                                          "Windows NT 10.0) AppleWebKit/537.36 ", "Mobile", "iPod touch",
#                                          "Android 6.0.1", "Windows NT 5.1) AppleWebKit/537.36 ", "Android 7.0",
#                                          "Windows Phone 10.0", "Windows NT 6.2) AppleWebKit/537.22 ", "Android 7.1.1",
#                                          "Windows NT 10.0.14393.1198"])
# # pd.options.display.mpl_style = 'default'
# scatter_matrix(data, alpha=0.2, figsize=(9, 9), diagonal='kde')
# data.groupby("gender").plot.hist()
# pyplot.show()
