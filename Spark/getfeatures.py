# coding=utf-8
import pymongo
import os
import io
import numpy

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
        weights = []
        contentid = []
        browser = []
        time = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
                16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
        ## fill weights
        # for clusternum in range(clusterscount):
        #     cluster = clusters[clusternum]
        #     weights.append(str(cluster.get("weight")))
        # weights.append(str(document.get("totalVisits")))
        # weights.append(str(document.get("rtlluVisits")))
        # weights.append(str(document.get("editusVisits")))
        # weights.append(str(document.get("habiterVisits")))

        ## fill timestamp
        for visits in document.get("lastVisits"):
            time[int(visits.get("timestamp")[11:13])] += 1
        towrite = gender + ',' + ','.join(weights) + ','.join(contentid) + ','.join(browser) + ','.join(
            list(map(str, time.values()))) + '\n'
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
print(len(gender))
from sklearn import svm
from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel='rbf')
scores = cross_val_score(clf, featurem, gender, cv=5)
print(scores)
