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
    allbrowsers = {}
    users_with_gender = users_collection.find({"gender": {"$ne": ""}})

    ##  get browser features
    dfeature1 = {}
    dfeature2 = {}
    dfeature3 = {}
    for d in users_with_gender:
        for visits in d.get("lastVisits"):
            deviceinfo = visits.get("browser").split('(')[1].split(')')[0].split(';')
            for dicount, di in enumerate(deviceinfo):
                if dicount == 0:
                    if di not in dfeature1.keys():
                        dfeature1[di] = 0
                elif dicount == 1:
                    if di not in dfeature2.keys():
                        dfeature2[di] = 0
                elif dicount == 3:
                    if di not in dfeature3.keys():
                        dfeature3[di] = 0

    users_with_gender = users_collection.find({"gender": {"$ne": ""}})
    for document in users_with_gender:
        gender = document.get("gender")
        clusters = document.get("clusters")
        clusterscount = 0
        weights1 = []
        weights2 = []
        timestamps = []
        dfeature1t = dfeature1
        dfeature2t = dfeature2
        dfeature3t = dfeature3
        for key in dfeature1t:
            dfeature1t[key] = 0
        for key in dfeature2t:
            dfeature2t[key] = 0
        for key in dfeature3t:
            dfeature3t[key] = 0
        timedataweekday = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                    15: 0,
                    16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
        timedataweekend = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                    15: 0,
                    16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
        clusteredtimedataweekday = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0, 'lnight': 0}
        clusteredtimedataweekend = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0, 'lnight': 0}
        weekday = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        devices = {"Macintosh": 0,
                   "iPhone": 0,
                   "Windows NT": 0,
                   "Linux": 0,
                   "iPad": 0,
                   "compatible": 0,
                   "Windows": 0,
                   "X11": 0,
                   "LTLM888 ": 0,
                   "Mobile": 0,
                   "iPod touch": 0,
                   "Android 6.0.1": 0,
                   "Android 5.1.1": 0,
                   "Android 7.0": 0,
                   "Windows Phone 10.0": 0,
                   "Android 7.1.1": 0}
        anotherdevices = {"Mobile": 0,
                          "iPhone": 0,
                          "SAMSUNG": 0,
                          "Windows": 0,
                          "Linux": 0,
                          "Mac OS": 0,
                          "iPod": 0,
                          "Tablet": 0,
                          "Moto": 0,
                          "LG": 0}
        # ## fill weights
        # for clusternum in range(clusterscount):
        #     cluster = clusters[clusternum]
        #     weights1.append(float(cluster.get("weight")))
        # if any(weights1):
        #     weights1 = [float(i) / sum(weights1) for i in weights1]

        # weights2.append(float(document.get("totalVisits")))
        weights2.append(float(document.get("rtlluVisits")))
        weights2.append(float(document.get("editusVisits")))
        weights2.append(float(document.get("habiterVisits")))
        weights2 = [float(i) / sum(weights2) for i in weights2]

        ## fill timestamp
        for visits in document.get("lastVisits"):
            dt = datetime.datetime.strptime(visits.get("timestamp")[0:10], "%Y-%m-%d")
            timestamps.append(datetime.datetime.strptime(visits.get("timestamp")[0:19], "%Y-%m-%dT%H:%M:%S"))
            weekday[int(dt.weekday())] += 1
            t = int(visits.get("timestamp")[11:13])
            if int(dt.weekday()) > 4:
                if (t >= 0) and (t < 5):
                    clusteredtimedataweekend['lnight'] += 1
                elif (t >= 5) and (t < 11):
                    clusteredtimedataweekend['morning'] += 1
                elif (t >= 11) and (t < 16):
                    clusteredtimedataweekend['afternoon'] += 1
                elif (t >= 16) and (t < 20):
                    clusteredtimedataweekend['evening'] += 1
                elif (t >= 20) and (t < 0):
                    clusteredtimedataweekend['night'] += 1
                timedataweekend[t] += 1
            else:
                if (t >= 0) and (t < 5):
                    clusteredtimedataweekday['lnight'] += 1
                elif (t >= 5) and (t < 11):
                    clusteredtimedataweekday['morning'] += 1
                elif (t >= 11) and (t < 16):
                    clusteredtimedataweekday['afternoon'] += 1
                elif (t >= 16) and (t < 20):
                    clusteredtimedataweekday['evening'] += 1
                elif (t >= 20) and (t < 0):
                    clusteredtimedataweekday['night'] += 1
                timedataweekday[t] += 1
        weekday = weekday.values()
        timedataweekday = timedataweekday.values()
        timedataweekend = timedataweekend.values()
        clusteredtimedataweekday = clusteredtimedataweekday.values()
        clusteredtimedataweekend = clusteredtimedataweekend.values()
        weekday = [float(i) / sum(weekday) for i in weekday]
        if any(timedataweekday):
            timedataweekday = [float(i) / sum(timedataweekday) for i in timedataweekday]
        if any(timedataweekend):
            timedataweekend = [float(i) / sum(timedataweekend) for i in timedataweekend]
        if any(clusteredtimedataweekday):
            clusteredtimedataweekday = [float(i) / sum(clusteredtimedataweekday) for i in clusteredtimedataweekday]
        if any(clusteredtimedataweekend):
            clusteredtimedataweekend = [float(i) / sum(clusteredtimedataweekend) for i in clusteredtimedataweekend]

        ## browser info
        for visits in document.get("lastVisits"):
            b = visits.get("browser").split('(')[1].split(';')[0]

            if visits.get("browser") in allbrowsers.keys():
                allbrowsers[visits.get("browser")] += 1
            else:
                allbrowsers[visits.get("browser")] = 1
            if "Windows NT" in b:
                devices["Windows NT"] += 1
            else:
                if b in devices.keys():
                    devices[b] += 1
        devices = devices.values()
        # devices = [float(i) / sum(devices) for i in devices]

        ## dfeature
        for visits in document.get("lastVisits"):
            deviceinfo = visits.get("browser").split('(')[1].split(')')[0].split(';')
            for dicount, di in enumerate(deviceinfo):
                try:
                    if di in dfeature1t.keys():
                        dfeature1t[di] += 1
                    elif di in dfeature2t.keys():
                        dfeature2t[di] += 1
                    elif di in dfeature3t.keys():
                        dfeature3t[di] += 1
                except:
                    pass
        dfeature1t = dfeature1t.values()
        if any(dfeature1t):
            dfeature1t = [float(i) / sum(dfeature1t) for i in dfeature1t]
        dfeature2t = dfeature2t.values()
        if any(dfeature2t):
            dfeature2t = [float(i) / sum(dfeature2t) for i in dfeature2t]
        dfeature3t = dfeature3t.values()
        if any(dfeature3t):
            dfeature3t = [float(i) / sum(dfeature3t) for i in dfeature3t]

        timespentinmin = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0,
                          14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}
        if len(timestamps) > 0:
            old = timestamps[0]
            for t in timestamps[1:]:
                totalsec = (t - old).total_seconds()
                if (totalsec < (10 * 60)) and (totalsec > 10):
                    timespentinmin[old.hour] += (totalsec / 60.0)
                old = t
        timespentinmin = timespentinmin.values()
        if any(timespentinmin):
            timespentinmin = [float(i) / sum(timespentinmin) for i in timespentinmin]

        # towrite = gender + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, weights2)) + ',' + ','.join(
        #     map(str, weekday)) + ',' + ','.join(map(str, timedata)) + '\n'
        # towrite = gender + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, devices)) + '\n'
        # towrite = gender + ',' + ','.join(map(str, weekday)) + '\n'
        # towrite = gender + ',' + str(list(devices).index(max(devices))) + ',' + str(list(timedataweekday).index(max(timedataweekday))) + '\n'
        # if len(document.get("lastVisits")) > 5:
        #     towrite = gender + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, devices)) + '\n'
        #     # towrite = gender + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, dfeature2t)) + '\n'
        #     featuresfile.write(towrite)
        # towrite = gender + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, devices)) + '\n'
        # featuresfile.write(towrite)
    for k in allbrowsers.keys():
        print(k, "->", allbrowsers[k])
    featuresfile.close()


features_file = "/tmp/rtleditus/genderpredictionfeatures2.dat"
extract_cluster_tofile(features_file=features_file)
# featuresmat = numpy.loadtxt(fname=features_file, delimiter=',', dtype=str)
# featuresmat[featuresmat == 'nan'] = '0.0'
# gender = featuresmat[:, 0]
# gender[gender == 'm'] = 1
# gender[gender == 'f'] = 0
# feature = featuresmat[:, 1:]
# featurem = feature[~(feature == '0.0')[:].all(1)]
# gender = gender[~(feature == '0.0')[:].all(1)]
# featurem = featurem.astype(float)
# gender = gender.astype(float)
#
# from sklearn import ensemble
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import KFold
# from sklearn.svm import SVC, LinearSVC
#
#
# n = []
# accuracy = []
# for nfeatures in range(len(featurem[0]), len(featurem[0])+1):
#     n.append(nfeatures)
#     n_folds = 5
#     # clf = ensemble.GradientBoostingClassifier(n_estimators=100)
#     # clf = SVC()
#     clf = LinearSVC()
#     print(featurem.shape, gender.shape)
#     from sklearn.feature_selection import RFE
#     rfe = RFE(clf, n_features_to_select=nfeatures)
#     rfe = rfe.fit(featurem, gender)
#     columnind = []
#     print(rfe.ranking_)
#     for i, s in enumerate(rfe.ranking_):
#         if s != 1:
#             columnind.append(i)
#     # print(rfe.support_)
#     # print(columnind)
#     # print(rfe.ranking_)
#     # print(rfe.score(featurem[3000:], gender[3000:]))
#     featuremnew = numpy.delete(featurem, columnind, axis=1)
#
#     kf = KFold(n_splits=n_folds)
#     original = []
#     predicted = []
#     for train_index, test_index in kf.split(featuremnew, gender):
#         X_train, X_test = featuremnew[train_index], featuremnew[test_index]
#         y_train, y_test = gender[train_index], gender[test_index]
#         clf.fit(X_train, y_train)
#         original = numpy.concatenate((original, y_test))
#         predicted = numpy.concatenate((predicted, clf.predict(X_test)))
#
#     tn, fp, fn, tp = confusion_matrix(original, predicted).ravel()
#     print(confusion_matrix(original, predicted))
#     print("Accuracy: ", (tn+tp)/(tn+fp+fn+tp))
#     accuracy.append((tn+tp)/(tn+fp+fn+tp))
# import matplotlib.pyplot as pyplot
# pyplot.plot(n, accuracy)
# pyplot.show()

# # pandas dataframe
# import pandas as pd
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as pyplot
#
# # names = ["gender", "Macintosh", "iPhone", "Windows NT", "Linux", "iPad",
# #          "compatible", "Windows", "X11",
# #          "LTLM888 ",
# #          "Mobile", "iPod touch",
# #          "Android 6.0.1", "Android 7.0",
# #          "Windows Phone 10.0", "Android 7.1.1"]
# data = pd.read_csv(features_file, names=range(16))
# data.replace(['m', 'f'], [1, 0], inplace=True)
# a = data[data[0]==0].mean()
# b = data[data[0]==1].mean()
# a.plot()
# b.plot()
# pyplot.show()
