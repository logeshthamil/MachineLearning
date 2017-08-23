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
        contentid = []
        browser = []
        timedata = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                    15: 0,
                    16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
        weekday = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
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

        towrite = gender + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, weights2)) + ',' + ','.join(
            map(str, weekday)) + ',' + ','.join(map(str, timedata)) + '\n'
        featuresfile.write(towrite)
    featuresfile.close()


features_file = "/tmp/rtleditus/genderpredictionfeatures2.dat"
extract_cluster_tofile(features_file=features_file)
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

from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import ensemble

n_folds = 5
clf = ensemble.GradientBoostingClassifier(n_estimators=100)
scores = cross_val_score(clf, featurem, gender, cv=n_folds)
print("Accuracy: ", numpy.mean(scores))
from sklearn.model_selection import KFold

kf = KFold(n_splits=n_folds)
original = []
predicted = []
for train_index, test_index in kf.split(featurem, gender):
    X_train, X_test = featurem[train_index], featurem[test_index]
    y_train, y_test = gender[train_index], gender[test_index]
    clf.fit(X_train, y_train)
    original = numpy.concatenate((original, y_test))
    predicted = numpy.concatenate((predicted, clf.predict(X_test)))

print(confusion_matrix(original, predicted))
