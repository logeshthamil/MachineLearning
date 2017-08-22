import os
import numpy
from sklearn.externals import joblib
import warnings
import optparse

path = os.path.dirname(os.path.realpath(__file__))
data = path + '/data/'
model = data + 'model/'
ids_file_name = 'all_cat_editus.dat'
ids_file = open(data + ids_file_name, 'r')
ids_file_txt = ids_file.read()
ids_txt_list = ids_file_txt.split('\n')
ids_txt_list.pop()
ids_file.close()

parser = optparse.OptionParser()
parser.add_option("--fn", dest="categories_id", type="str", help="get the editus categories")
(options, args) = parser.parse_args()
p_file_name = options.categories_id

warnings.filterwarnings("ignore", category=DeprecationWarning)
model1 = joblib.load(model + 'editus_parameters_GradientBoostingClassifier.pkl')
# model2 = joblib.load(model + 'editus_parameters_BaggingClassifier.pkl')
# model3 = joblib.load(model + 'editus_parameters_ExtraTreesClassifier.pkl')
# model4 = joblib.load(model + 'editus_parameters_SVC.pkl')
p_file = open(p_file_name, 'r')
for line in p_file:
    g = []
    param = numpy.zeros(len(ids_txt_list))
    if line != "":
        l = line.split(' ')
        if l[0] != "":
            ids = l[0].split(',')
        for i in ids:
            try:
                ind = ids_txt_list.index(i)
                param[ind] = param[ind] + 1  # for tf input
                # param[ind] = 1 #for boolean input
            except:
                pass
    # print param
    p1 = model1.predict(param)
    print p1[0]
p_file.close()
