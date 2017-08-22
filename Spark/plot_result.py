import os, numpy
from sklearn.externals import joblib
import sklearn.dummy
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


def get_all_models(keyword=None):
    """
    Get all the model files with the keyword and load it as model and returns the model
    :param keyword: the keyword which has to be found in the model file
    :return: all the the models in an array
    """
    models = []
    folder_path = os.path.realpath(__file__)[:-12]
    data_directory = folder_path + "data/"
    model_directory = data_directory + "model/"
    model_directory = "/home/lt/quanox/QX_Recommendations/recommendation-python/Profiling_for_rtl/gender_prediction_onlyeditus/data/model/"
    for model_dir in os.listdir(model_directory):
        if keyword in model_dir:
            model = joblib.load(model_directory + model_dir)
            models.append(model)
    return models


def GetAccuracyofsavedmodels():
    gender_path = "/home/lt/quanox/QX_Recommendations/recommendation-python/Profiling_for_rtl/gender_prediction/data/gender.dat"
    param_matrix = "/home/lt/quanox/QX_Recommendations/recommendation-python/Profiling_for_rtl/gender_prediction/data/editus_parameters.dat"
    models = get_all_models(keyword="editus_parameters")
    y_digits = numpy.loadtxt(gender_path, dtype=float)
    X_digits = numpy.loadtxt(param_matrix, dtype=float)
    dummy = sklearn.dummy.DummyClassifier()
    dummy.fit(X_digits, y_digits)
    outputs = []
    models.append(dummy)
    for classifier in models:
        print 'name of the classifier: ' + str(classifier)
        predicted = classifier.predict(X_digits)
        outputs.append(predicted)
        original = y_digits
        # plot_histogram_normal(error_list=original - predicted)
        print 'score of the classifier: %f' % classifier.score(X_digits, y_digits)
        print '\n \n'
    from scipy.stats import mode
    most_frequent = mode(outputs)
    out = most_frequent[0][0]
    o = []
    for a, b in zip(out, y_digits):
        if int(a) == int(b):
            o.append("success")
    print "Accuracy ensemble: " + str((len(o) / float(len(y_digits))) * 100.0)


def compute_accuracy_and_plot_roc():
    gender_path = "/home/lt/quanox/QX_Recommendations/recommendation-python/Profiling_for_rtl/gender_prediction_onlyeditus/data/gender.dat"
    param_matrix = "/home/lt/quanox/QX_Recommendations/recommendation-python/Profiling_for_rtl/gender_prediction_onlyeditus/data/editus_parameters.dat"
    models = get_all_models(keyword="editus_parameters")
    y_digits = numpy.loadtxt(gender_path, dtype=float)
    X_digits = numpy.loadtxt(param_matrix, dtype=float)
    X = X_digits[2500:]
    y = y_digits[2500:]
    print X.shape
    print y.shape

    for classifier in models:
        print classifier
        predicted = classifier.predict(X)
        girls = []
        boys = []
        acc = 0
        for p, o in zip(predicted, y):
            if p == 0:
                girls.append(p)
            else:
                boys.append(p)
            if p == o:
                acc = acc + 1
        print "Accuracy: " + str((acc / float(len(y))) * 100)
        print len(boys)
        print len(girls)
        print

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    colors = cycle(['blue', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2

    i = 0
    cv = StratifiedKFold(n_splits=6)
    classifier = models[1]
    mean_tpr1 = mean_tpr
    mean_fpr1 = mean_fpr
    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = classifier.predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr1, tpr1, thresholds1 = roc_curve(y[test], y[test])
        mean_tpr1 += interp(mean_fpr1, fpr1, tpr1)
        mean_tpr1[0] = 0.0
        roc_auc1 = auc(fpr1, tpr1)
        plt.plot(fpr1, tpr1, lw=lw - 0.75, linestyle='--', color=color,
                 label='Perfect predictor (area = 1.0)')

        i += 1
        break

    for (train, test), color in zip(cv.split(X, y), colors):
        probas_ = classifier.predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        # plt.plot(fpr, tpr, lw=lw, color=color,
        #          label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='-',
             label='QuanoX (area = %0.1f)' % mean_auc, lw=lw + 2)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw - 0.75, color='r',
             label='Random (area = 0.5)')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Gender prediction')
    plt.legend(loc="lower right")
    plt.show()


compute_accuracy_and_plot_roc()
