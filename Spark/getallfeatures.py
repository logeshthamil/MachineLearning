# coding=utf-8
import pymongo
import os
import io
import numpy
import datetime, random

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
        timestamps = []
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
            timestamps.append(datetime.datetime.strptime(visits.get("timestamp")[0:19], "%Y-%m-%dT%H:%M:%S"))
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

        if (age is not None) and (len(document.get("lastVisits")) > 5):
            towrite = str(age) + ',' + ','.join(map(str, devices)) + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, weights2)) + '\n'
            # towrite = str(age) + ',' + ','.join(map(str, weights1)) + ',' + ','.join(map(str, timedata)) + ',' + ','.join(map(str, devices)) + ',' + ','.join(map(str, weekday)) + '\n'
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
# import matplotlib.pyplot as pyplot
# pyplot.hist(age, bins=range(1, 81))
# pyplot.show()
af = {}
for a, f in zip(age, featurem):
    if a not in af.keys():
        af[a] = []
    af[a].append(f.tolist())
mfeaturem = []
mage = []
for k, v in af.items():
    for mv in v[:50]:
        mfeaturem.append(mv)
        mage.append(k)
c = list(zip(mfeaturem, mage))
random.shuffle(c)
featurem, age = zip(*c)
import matplotlib.pyplot as pyplot
pyplot.hist(age, bins=range(1, 81))
# pyplot.show()
featurem = numpy.asarray(featurem)
age = numpy.asarray(age)

from sklearn import svm, ensemble
from sklearn.model_selection import KFold
random_forest = ensemble.RandomForestRegressor(n_estimators=100)
gradient_boost = ensemble.GradientBoostingRegressor(n_estimators=1000)
svm_ = svm.SVR(kernel='linear')
svm_poly = svm.SVR(kernel='poly', degree=3)
svm_rbf = svm.SVR(kernel='rbf', degree=5)
n_folds = 5

regres = gradient_boost
kf = KFold(n_splits=n_folds)
original = []
predicted = []
for train_index, test_index in kf.split(featurem, age):
    X_train, X_test = featurem[train_index], featurem[test_index]
    y_train, y_test = age[train_index], age[test_index]
    regres.fit(X_train, y_train)
    original = numpy.concatenate((original, y_test))
    predicted = numpy.concatenate((predicted, regres.predict(X_test)))

pyplot.hist(predicted, bins=range(1, 81))
pyplot.show()
# for p,o in zip(predicted, original):
#     print(p, ' -> ', o)
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





# n = []
# accuracy = []
# for nfeatures in range(1, len(featurem[0])+1, 2):
#     n.append(nfeatures)
#     n_folds = 5
#     # clf = ensemble.GradientBoostingClassifier(n_estimators=100)
#     # clf = SVC()
#     clf = regres
#     print(featurem.shape, age.shape)
#     from sklearn.feature_selection import RFE
#     rfe = RFE(clf, n_features_to_select=nfeatures)
#     rfe = rfe.fit(featurem, age)
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
#     for train_index, test_index in kf.split(featuremnew, age):
#         X_train, X_test = featuremnew[train_index], featuremnew[test_index]
#         y_train, y_test = age[train_index], age[test_index]
#         clf.fit(X_train, y_train)
#         original = numpy.concatenate((original, y_test))
#         predicted = numpy.concatenate((predicted, clf.predict(X_test)))
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     ax.scatter(original, predicted, edgecolors=(0, 0, 0))
#     ax.plot([10, 80], [10, 80], 'k--', lw=4)
#     ax.set_xlabel('Measured')
#     ax.set_ylabel('Predicted')
#     plt.show()
#
#     from sklearn.metrics import mean_absolute_error
#     accuracy.append(mean_absolute_error(original, predicted))
#     print("Mean Squared Error", mean_absolute_error(original, predicted))
#
# import matplotlib.pyplot as pyplot
# pyplot.plot(n, accuracy)
# pyplot.show()
