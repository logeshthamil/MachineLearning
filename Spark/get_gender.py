import os, optparse, io, numpy, warnings
from sklearn.externals import joblib
from scipy.stats import mode
from pymongo import MongoClient

warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_all_models(keyword=None):
    """
    Get all the model files with the keyword and load it as model and returns the model
    :param keyword: the keyword which has to be found in the model file
    :return: all the the models in an array
    """
    models = []
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    model_directory = data_directory + "model/"
    for model_dir in os.listdir(model_directory):
        if keyword in model_dir:
            model = joblib.load(model_directory + model_dir)
            models.append(model)
    return models


def GetEditusCategory(_id=None, depth=3):
    """
    Return the id by getting the categories of certain depth.
    :param depth: the depth of the categories
    """
    client = MongoClient('mongodb://qxuser:userqx16@52.57.103.28/?authSource=users')
    __all_editus_categories = client.RtlDB.EditusCategs
    lvl = int(__all_editus_categories.find({'id': _id})[0].get('lvl'))
    # print c, lvl, type(lvl), depth, type(depth)
    if lvl == depth:
        return _id
    elif lvl > depth:
        while lvl != depth:
            category = __all_editus_categories.find({'id': c})[0].get('parent')
            lvl = int(__all_editus_categories.find({'category': category})[0].get('lvl'))
            c = __all_editus_categories.find({'category': category})[0].get('id')
            print c, lvl, depth, category
        return c
    else:
        return None


def get_gender_usingtf_testaccuracy():
    """
    A function to test the accuracy of the gender prediction with the training set.
    """
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    tf_input_file = data_directory + "tf.dat"
    gender_output_file = data_directory + "gender.dat"
    all_tf_file = data_directory + "all_cat_tf.dat"
    _lambada_file = data_directory + "lda_outputs/lambda.dat"
    all_users_gender = map(lambda s: s.strip(), io.open(gender_output_file).readlines())
    tf_models = get_all_models(keyword="tf_parameters")
    tf_models = [tf_models[0]]
    all_tf = io.open(all_tf_file, encoding='utf-8').readlines()
    tf_inputs = io.open(tf_input_file).readlines()
    all_tf_words = []
    lambda_ = numpy.transpose(numpy.loadtxt(_lambada_file))
    for tf in all_tf:
        all_tf_words.append(tf.split('\t')[1])
    all_users_topic_proportions = []
    output = []
    for index, tf_input in enumerate(tf_inputs):
        topic_proportions = []
        input_tfs = tf_input.split(' ')
        input_tfs = map(lambda s: s.strip(), input_tfs)
        for input_tf in input_tfs:
            try:
                ind = all_tf_words.index(input_tf)
                norm = [float(i) / sum(lambda_[ind]) for i in lambda_[ind]]
                topic_proportions.append(norm)
            except:
                print "exception"
                pass
        all_users_topic_proportions.append(numpy.sum(numpy.asarray(topic_proportions), axis=0))
        try:
            p_s = []
            for tf_model in tf_models:
                p = int(tf_model.predict(numpy.sum(numpy.asarray(topic_proportions), axis=0).astype(list))[0])
                p_s.append(p)
            most_frequent = mode(p_s)
            p = most_frequent[0][0]
        except:
            print "Exception"
            p = 1
        if p == int(all_users_gender[index]):
            output.append('success')
        else:
            print "Failure"
    print "Accuracy of gender prediction: " + str((len(output) / float(len(all_users_gender))) * 100.0)


def get_gender_usingeditus_testaccuracy():
    """
    A function to test the accuracy of the gender prediction with the training set.
    """
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    editus_input_file = data_directory + "editus_categories.dat"
    gender_output_file = data_directory + "gender.dat"
    all_editus_file = data_directory + "all_cat_editus.dat"
    all_users_gender = map(lambda s: s.strip(), io.open(gender_output_file).readlines())
    editus_models = get_all_models(keyword="editus_parameters")
    all_editus = io.open(all_editus_file, encoding='utf-8').readlines()
    all_editus_words = map(lambda s: s.strip(), all_editus)
    editus_inputs = io.open(editus_input_file).readlines()
    output = []
    for index, editus_input in enumerate(editus_inputs):
        param = numpy.zeros(len(all_editus_words))
        input_edituss = editus_input.split(' ')
        input_edituss = map(lambda s: s.strip(), input_edituss)
        for input_editus in input_edituss:
            try:
                input_editus = GetEditusCategory(_id=input_editus)
                ind = all_editus_words.index(input_editus)
                param[ind] = param[ind] + 1
            except:
                pass
        try:
            p_s = []
            for editus_model in editus_models:
                p = int(editus_model.predict(param))
                p_s.append(p)
            most_frequent = mode(p_s)
            p = most_frequent[0][0]
        except:
            p = 1
            print "exception"
        # print p
        if p == int(all_users_gender[index]):
            output.append('success')
            print "Success"
        else:
            print "Failure"
    print "Accuracy of gender prediction: " + str((len(output) / float(len(all_users_gender))) * 100.0)


def get_gender_usingrtl_testaccuracy():
    """
    A function to test the accuracy of the gender prediction with the training set.
    """
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    rtl_input_file = data_directory + "rtl_categories.dat"
    gender_output_file = data_directory + "gender.dat"
    all_rtl_file = data_directory + "all_cat_rtl.dat"
    all_users_gender = map(lambda s: s.strip(), io.open(gender_output_file).readlines())
    rtl_models = get_all_models(keyword="rtl_parameters")
    all_rtl = io.open(all_rtl_file, encoding='utf-8').readlines()
    all_rtl_words = map(lambda s: s.strip(), all_rtl)
    rtl_inputs = io.open(rtl_input_file).readlines()
    output = []
    for index, rtl_input in enumerate(rtl_inputs):
        param = numpy.zeros(len(all_rtl_words))
        input_rtls = rtl_input.split(' ')
        input_rtls = map(lambda s: s.strip(), input_rtls)
        for input_rtl in input_rtls:
            try:
                ind = all_rtl_words.index(input_rtl)
                param[ind] = param[ind] + 1
            except:
                pass
        try:
            p_s = []
            for rtl_model in rtl_models:
                p = int(rtl_model.predict(param))
                p_s.append(p)
            most_frequent = mode(p_s)
            p = most_frequent[0][0]
        except:
            p = 1
            print "exception"
        print p
        if p == int(all_users_gender[index]):
            output.append('success')
            print "Success"
        else:
            print "Failure"
    print "Accuracy of gender prediction: " + str((len(output) / float(len(all_users_gender))) * 100.0)


def get_gender_usingtime_testaccuracy():
    """
    A function to test the accuracy of the gender prediction with the training set.
    """
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    time_input_file = data_directory + "time_stamp.dat"
    gender_output_file = data_directory + "gender.dat"
    all_time_file = data_directory + "all_cat_time.dat"
    all_users_gender = map(lambda s: s.strip(), io.open(gender_output_file).readlines())
    time_models = get_all_models(keyword="time_parameters")
    all_time = io.open(all_time_file, encoding='utf-8').readlines()
    all_time_words = map(lambda s: s.strip(), all_time)
    print time_input_file
    time_inputs = io.open(time_input_file).readlines()
    output = []
    for index, time_input in enumerate(time_inputs):
        param = numpy.zeros(len(all_time_words))
        input_times = time_input.split(' ')
        input_times = map(lambda s: s.strip(), input_times)
        for input_time in input_times:
            try:
                ind = all_time_words.index(input_time)
                param[ind] = param[ind] + 1
            except:
                pass
        try:
            p_s = []
            for time_model in time_models:
                p = int(time_model.predict(param))
                p_s.append(p)
            most_frequent = mode(p_s)
            p = most_frequent[0][0]
        except:
            p = 1
            print "exception"
        print p
        if p == int(all_users_gender[index]):
            output.append('success')
            print "Success"
        else:
            print "Failure"
    print "Accuracy of gender prediction: " + str((len(output) / float(len(all_users_gender))) * 100.0)


def get_output_editus(models, data_file):
    """
    Give the data to the models and get the output. Also combine the outputs and give it in an array
    :param models: the sklearn models for age or gender prediction
    :param data: the data which have to be fed to the model
    :return: a list of outputs
    """
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    all_editus_file = data_directory + "all_cat_editus.dat"
    editus_models = models
    all_editus_words = map(lambda s: s.strip(), io.open(all_editus_file, encoding='utf-8').readlines())
    with open(data_file) as f:
        for line in f:
            param = numpy.zeros(len(all_editus_words)).astype(int)
            input_editus = line.split(' ')[0]
            input_param = input_editus.split(',')
            for ie in input_param:
                try:
                    # input_editus = GetEditusCategory(_id=input_editus)
                    ind = all_editus_words.index(ie)
                    param[ind] = param[ind] + 1
                except:
                    pass
            p_s = []
            for editus_model in editus_models:
                p = editus_model.predict(param)
                p_s.append(p)
            most_frequent = mode(p_s)
            p = most_frequent[0][0]
            print int(p[0])


def get_output_rtl(models, data_file):
    """
    Give the data to the models and get the output. Also combine the outputs and give it in an array
    :param models: the sklearn models for age or gender prediction
    :param data: the data which have to be fed to the model
    :return: a list of outputs
    """
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    all_rtl_file = data_directory + "all_cat_rtl.dat"
    rtl_models = models
    all_rtl_words = map(lambda s: s.strip(), io.open(all_rtl_file, encoding='utf-8').readlines())
    with open(data_file) as f:
        for line in f:
            param = numpy.zeros(len(all_rtl_words)).astype(int)
            input_rtl = line.split(' ')[1]
            input_param = input_rtl.split(',')
            for ie in input_param:
                try:
                    # input_rtl = GetEditusCategory(_id=input_rtl)
                    ind = all_rtl_words.index(ie)
                    param[ind] = param[ind] + 1
                except:
                    pass
            p_s = []
            for rtl_model in rtl_models:
                p = rtl_model.predict(param)
                p_s.append(p)
            most_frequent = mode(p_s)
            p = most_frequent[0][0]
            print int(p[0])


def get_output_time(models, data_file):
    """
    Give the data to the models and get the output. Also combine the outputs and give it in an array
    :param models: the sklearn models for age or gender prediction
    :param data: the data which have to be fed to the model
    :return: a list of outputs
    """
    folder_path = os.path.realpath(__file__)[:-13]
    data_directory = folder_path + "data/"
    all_time_file = data_directory + "all_cat_time.dat"
    time_models = models
    all_time_words = map(lambda s: s.strip(), io.open(all_time_file, encoding='utf-8').readlines())
    with open(data_file) as f:
        for line in f:
            param = numpy.zeros(len(all_time_words)).astype(int)
            input_time = line.split(' ')[2]
            input_param = input_time.split(',')
            for ie in input_param:
                try:
                    # input_time = GetEditusCategory(_id=input_time)
                    ind = all_time_words.index(ie)
                    param[ind] = param[ind] + 1
                except:
                    pass
            p_s = []
            for time_model in time_models:
                p = time_model.predict(param)
                p_s.append(p)
            most_frequent = mode(p_s)
            p = most_frequent[0][0]
            print p[0]


def main():
    parser = optparse.OptionParser()
    parser.add_option("--fn", dest="file_name", type="str", help="get the filename in which the inputs are stored")
    parser.add_option("--c", dest="choose", type="str", help="choose the feature vector to predict the gender")
    parser.add_option("--testalleditus", dest="test_all_editus", type="int", default=0,
                      help="test the model using the training set")
    parser.add_option("--testallrtl", dest="test_all_rtl", type="int", default=0,
                      help="test the model using the training set")
    parser.add_option("--testalltime", dest="test_all_time", type="int", default=0,
                      help="test the model using the training set")
    (options, args) = parser.parse_args()
    test_all_e = options.test_all_editus
    test_all_r = options.test_all_rtl
    test_all_t = options.test_all_time
    outputs = []
    if test_all_e == 1:
        get_gender_usingeditus_testaccuracy()
    if test_all_r == 1:
        get_gender_usingrtl_testaccuracy()
    if test_all_t == 1:
        get_gender_usingtime_testaccuracy()
    if options.file_name and options.choose is not None:
        c = options.choose
        cat = [int(x) for x in c.split(',')]
        for c in cat:
            if c == 0:
                ec_models = get_all_models(keyword="editus_parameters")
                o = get_output_editus(ec_models, options.file_name)
                outputs.append(o)
            elif c == 1:
                rc_models = get_all_models(keyword="rtl_parameters")
                o = get_output_rtl(rc_models, options.file_name)
                outputs.append(o)
            elif c == 2:
                tf_models = get_all_models(keyword="time_parameters")
                o = get_output_time(tf_models, options.file_name)
                outputs.append(o)


if __name__ == "__main__":
    main()
