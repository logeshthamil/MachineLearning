# -*- coding: utf-8 -*-
from pymongo import MongoClient
import io, numpy, string, warnings, time, shutil, subprocess, os, datetime
from sklearn import svm, preprocessing, ensemble
from sklearn.gaussian_process import GaussianProcessClassifier
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import norm
from sklearn.externals import joblib


def write_numpynd_to_file(numpy_array=None, file_location=None):
    """
    Write the numpy array to a text file.

    :param numpy_array: the input numpy array
    :param file_location: the file location
    :return: None
    """
    numpy.savetxt(fname=file_location, X=numpy_array)


def output_to_desired(doc_topic_location=None, topic_word_location=None, total_topic_location=None):
    """
    Transform the output of lda which is unnormalized into the desired format with normalized values.

    :param doc_topic_location: output location where doc topic is saved
    :param topic_word_location: output location where topic word is saved
    :param total_topic_location: output location where the total count of documents in topics is saved
    :return: lambda and gamma
    """
    warnings.filterwarnings("ignore")
    doc_topic = numpy.loadtxt(doc_topic_location, delimiter="\n", dtype=str)
    topic_word = numpy.loadtxt(topic_word_location, delimiter="\n", dtype=str)
    total_topic = numpy.loadtxt(total_topic_location, delimiter=" ", dtype=str)[1:]
    no_of_topics = len(total_topic)
    no_of_docs = len(doc_topic)
    no_of_words = len(topic_word)
    doc_topic_numpy = numpy.zeros((no_of_docs, no_of_topics))
    topic_word_numpy = numpy.zeros((no_of_topics, no_of_words))
    for doc_number, i_chunk in enumerate(doc_topic):
        i_chunk = i_chunk.split(" ")[2:]
        for i_i_chunk in i_chunk:
            topic, weight = i_i_chunk.split(":")
            doc_topic_numpy[doc_number, topic] = int(weight)
    for word_number, i_word in enumerate(topic_word):
        i_word = i_word.split(" ")[1:]
        for i_i_word in i_word:
            topic, weight = i_i_word.split(":")
            topic_word_numpy[topic, word_number] = int(weight)

    # normalize
    # doc_topic_numpy_norm = normalize(doc_topic_numpy, norm='l1', axis=1)
    # topic_word_numpy_norm = normalize(topic_word_numpy, norm='l1', axis=0)

    # dont normalize
    doc_topic_numpy_norm = doc_topic_numpy
    topic_word_numpy_norm = topic_word_numpy

    # replace zero value with minimum value
    # doc_topic_numpy_norm[doc_topic_numpy_norm == 0] = 1 ** -15
    # topic_word_numpy_norm[topic_word_numpy_norm == 0] = 1 ** -15

    return doc_topic_numpy_norm, topic_word_numpy_norm


def check_vocab(input_word):
    """
    return False if special char (defined) are found in input_word
    return True otherwise.
    A function to check whether the vocabulary is valid or not.
    """
    # ... replace("char you allow","char you don t allow")
    invalid_chars = set(string.punctuation.replace("", "’ -'´`"))
    if any(char in invalid_chars for char in input_word):
        return False
    elif len(input_word) < 5:
        return False
    else:
        return True


def transform_lightlda_outputs(doc_topic=None, topic_word=None, total_topic=None, doc_topic_location=None,
                               topic_word_location=None, save=False):
    """
    Transform the output of the light lda (c++ version) to readable and universal format.

    :param doc_topic: the doc topic probablity distribution file
    :param topic_word: the toipic word probablity distribution file
    :param total_topic: the total topic count file
    :param doc_topic_location: the normalized doc topic probablity distribution file
    :param topic_word_location: the normalized topic word probablity distribution file
    :param save: if False return the gamma and lambda array, else save the gamma and lambda array in the file
    """
    _gamma, _lambda = output_to_desired(doc_topic_location=doc_topic,
                                        topic_word_location=topic_word,
                                        total_topic_location=total_topic)
    if save is True:
        write_numpynd_to_file(numpy_array=_gamma, file_location=doc_topic_location)
        write_numpynd_to_file(numpy_array=_lambda, file_location=topic_word_location)
    else:
        return _gamma, _lambda


def _transform_corpusdata_to_libsvm(_vocab_full_location, _corpus_location, __transformed_corpus, __transformed_vocab):
    """
    A method to transform the corpus from standard format to libsvm format to use it as an input for lightLDA.
    """
    # generating the input corpus file for light lda
    print "converting the standard corpus to libsvm format to use it as input for light lda"

    vocab_location = io.open(_vocab_full_location, mode='r').readlines()
    vocab = map(lambda s: s.strip(), vocab_location)
    _len_vocab = len(vocab)
    corpus_location = io.open(_corpus_location, mode='r').readlines()
    corpus = map(lambda s: s.strip(), corpus_location)
    _len_corpus = len(corpus)
    transformed_corpus = io.open(__transformed_corpus, mode='w')
    for inde, i_corpus in enumerate(corpus):
        inde = str(inde) + '\t'
        transformed_corpus.write(str(inde).decode('unicode-escape'))
        corpus_vocabs = i_corpus.split(' ')
        index = {}
        for i_corpus_vocabs in corpus_vocabs:
            try:
                i = vocab.index(i_corpus_vocabs)
                if i in index.keys():
                    index[i] += 1
                else:
                    index.update({i: 1})
            except:
                print "exceptional word: " + str(i_corpus_vocabs)
        ind = 0
        for i in index:
            temp = str(i) + ":" + str(index[i]) + " "
            transformed_corpus.write(temp.decode('unicode-escape'))
            ind = ind + 1
        transformed_corpus.write(u'\n')
    transformed_corpus.close()

    # generating the word index file for light lda
    print "Generating the word index file to use it as input for light lda" + "\n"
    transformed_vocab = io.open(__transformed_vocab, mode='w')
    count = {}
    for w in io.open(_corpus_location).read().split():
        if w in count:
            count[w] += 1
        else:
            count[w] = 1
    for i, c in enumerate(vocab):
        t = str(i) + '\t'
        transformed_vocab.write(t.decode('unicode-escape'))
        transformed_vocab.write(c)
        t = '\t' + str(count[c]) + '\n'
        transformed_vocab.write(t.decode('unicode-escape'))


def plot_histogram_normal(error_list=None):
    """
    A function to plot the histogram from the error array.
    :param error_list: a list of the errors
    """
    (mu, sigma) = norm.fit(error_list)
    n, bins, patches = plt.hist(error_list, 40, facecolor='green', alpha=0.75, normed=1)  # , normed=1
    y = mlab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=2)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    white_patch = mpatches.Patch(color='white', label=r'$\mathrm{}\mu=%0.2f$' % mu)
    white_patch2 = mpatches.Patch(color='white', label=r'$\mathrm{}\sigma=%0.2f$' % sigma)
    plt.legend(handles=[white_patch, white_patch2])
    plt.show()


class GenderPrediction_rtlcategories(object):
    """
    A class to predict the age of the users.
    """

    def __init__(self, total_number_of_items=None, total_number_of_users=None,
                 uri='mongodb://qxuser:userqx16@52.57.103.28/?authSource=users', data_path=None, model_directory=None):
        """
        :param total_number_of_items: the total number of items which has to be extracted
        :param total_number_of_users: the total number of users to be extracted
        :param uri: the uri from which the datas can be extracted
        :param data_path: the path in which the data can be stored
        """
        # declare the input parameters
        self.__no_of_items = total_number_of_items
        self.__no_of_users = total_number_of_users
        client = MongoClient(uri)
        self.__user_collection = client.RtlDB.Users2
        self.__article_collection = client.RtlDB.Articles2
        self.__all_rtl_categories = client.RtlDB.RtlCategs
        self.__all_editus_categories = client.RtlDB.EditusCategs
        if self.__no_of_users is None:
            self.__no_of_users = self.__user_collection.find().count()
        if self.__no_of_items is None:
            self.__no_of_items = self.__article_collection.find().count()
        query = {"$and": [{"gender": {"$ne": ''}}, {"articles": {"$ne": []}}]}
        self.__all_users = self.__user_collection.find(query, no_cursor_timeout=True).limit(self.__no_of_users)
        self.__all_items = self.__article_collection.find().limit(self.__no_of_items)
        self.__data_path = data_path
        self.__model_path = model_directory
        self.__user_id_path = data_path + 'userid.dat'
        self.__rtl_categories_path = data_path + 'rtl_categories.dat'
        self.__editus_categories_path = data_path + 'editus_categories.dat'
        self.__time_stamp_path = data_path + 'time_stamp.dat'
        self.__tf_path = data_path + "tf.dat"
        self.__gender_path = data_path + 'gender.dat'
        self.__time_param_matrix = self.__data_path + "time_parameters.dat"
        self.__rtl_param_matrix = self.__data_path + "rtl_parameters.dat"
        self.__editus_param_matrix = self.__data_path + "editus_parameters.dat"
        self.__tf_param_matrix = self.__data_path + "tf_parameters.dat"
        self.__all_categories_time = self.__data_path + 'all_cat_time.dat'
        self.__all_categories_tf = self.__data_path + 'all_cat_tf.dat'
        self.__all_categories_rtl = self.__data_path + 'all_cat_rtl.dat'
        self.__all_categories_editus = self.__data_path + 'all_cat_editus.dat'
        self.__age_path = data_path + 'age.dat'

    def Getagefromyear(self, year=None):
        """
        Assign a class label for the year label
        :param year: the year in which the user is born
        :return: the class label
        """
        if year is None:
            print "Please enter the year to assign class to them"
        t = datetime.datetime.today()
        b = datetime.datetime.strptime(str(year), '%Y')
        a = (t - b).days / 365
        return str(a)

    def _Construct_Matrix(self, xlabels=None, entries=None, output_matrix_location=None,
                          output_ylabels_location=None, binary_matrix=False, order_labels=False):
        """
        Construct matrix from the data file.
        :param xlabels: the xlabels file
        :param entries: the data file
        :param output_matrix_location: the output matrix location file
        :param output_ylabels_location: the output y label location file
        :param binary_matrix: if true only binary elements are inserted into the matrix
        """
        print "Constructing matrix from the data" + "\n"
        xlabels = io.open(xlabels).readlines()
        entries = io.open(entries).readlines()
        u_entries = []
        for i in entries:
            t = i.split(' ')
            t[-1] = t[-1].strip()
            u_entries.append(t)
        unique_entries = list(set(sum(u_entries, [])))
        output_matrix = numpy.zeros((len(xlabels), len(unique_entries))).astype(int)
        output_ylabels = []
        print "Shape of output matrix :" + str(output_matrix.shape)
        for x_ind, entry in zip(range(len(xlabels)), entries):
            t = entry.split(' ')
            t[-1] = t[-1].strip()
            for ent in t:
                if ent not in output_ylabels:
                    output_ylabels.append(ent)
                y_ind = output_ylabels.index(ent)
                output_matrix[x_ind, y_ind] = output_matrix[x_ind, y_ind] + 1
        if binary_matrix is True:
            output_matrix = (output_matrix != 0).astype(int)
        if order_labels is True:
            order = map(int, output_ylabels)
            temp_dict = {}
            output_matrix_temp = []
            for ind, values in zip(order, numpy.transpose(output_matrix)):
                temp_dict.update({ind: values})
            for dic in temp_dict:
                output_matrix_temp.append(temp_dict[dic])
            output_matrix = numpy.transpose(numpy.asarray(output_matrix_temp))
            output_ylabels = sorted(order)
        # output_matrix[output_matrix == 0] = 0.00000001
        # output_matrix = preprocessing.normalize(output_matrix, norm='l1', axis=1)
        # output_matrix[output_matrix <= 0.05] = 0
        numpy.savetxt(io.open(output_matrix_location, mode='wb'), output_matrix.astype(int))
        with io.open(output_ylabels_location, mode='w', encoding='utf-8') as oml:
            for ylabel in output_ylabels:
                if type(ylabel) is int:
                    ylabel = str(ylabel).decode('utf-8')
                oml.write(ylabel)
                oml.write(u'\n')
        oml.close()

    def GetCategory(self, _id=None, depth=2):
        """
        Return the id by getting the categories of certain depth.
        :param depth: the depth of the categories
        """
        lvl = int(self.__all_editus_categories.find({'id': _id})[0].get('lvl'))
        # print c, lvl, type(lvl), depth, type(depth)
        if lvl == depth:
            return _id
        elif lvl > depth:
            print "level greater than depth"
            print _id, lvl, depth, self.__all_rtl_categories.find({'id': _id})[0].get('category')
            while lvl != depth:
                category = self.__all_editus_categories.find({'id': c})[0].get('parent')
                lvl = int(self.__all_editus_categories.find({'category': category})[0].get('lvl'))
                c = self.__all_editus_categories.find({'category': category})[0].get('id')
                print c, lvl, depth, category
            print
            return c
        else:
            return None

    def GetAllData2(self):
        """
        Get all the data which can be used for gender prediction
        """
        class_file = io.open(self.__gender_path, mode='ab+')
        user_file = io.open(self.__user_id_path, mode='ab+')
        editus_categories_file = io.open(self.__editus_categories_path, mode='ab+')
        self.__class_labels = []
        print "Extracting the categories of the valid users" + "\n"
        for i in self.__all_users:
            try:
                editus_category_ids = []
                u_id = i.get('id')
                total_tfs = []
                articles = i.get('articles')
                gender = i.get('gender')
                if gender == 'm':
                    gender = '1'
                elif gender == 'f':
                    gender = '0'
                else:
                    break
                s_articles = sorted(articles, key=lambda k: k['timestamp'])
                for i_a, article in enumerate(s_articles):
                    a_id = article.get('id')
                    art = self.__article_collection.find({'id': a_id})[0]
                    editus_categories = art.get('editusCategs')
                    # check if the tf vector is empty
                    if editus_categories:
                        for r_c in editus_categories:
                            c_id = self.__all_editus_categories.find({'category': r_c})[0].get('id')
                            if c_id is not None:
                                editus_category_ids.append(c_id)
                if len(editus_category_ids) > 0:
                    print u_id, editus_category_ids
                    e = ' '.join(editus_category_ids)
                    user_file.write(u_id + '\n')
                    class_file.write(gender + '\n')
                    editus_categories_file.write(e + '\n')
            except Exception as e:
                print str(e) + '\n'
        user_file.close()
        class_file.close()
        editus_categories_file.close()

    def ConstructParameterMatrix(self):
        self._Construct_Matrix(xlabels=self.__gender_path, entries=self.__editus_categories_path,
                               output_matrix_location=self.__editus_param_matrix,
                               output_ylabels_location=self.__all_categories_editus,
                               binary_matrix=False, order_labels=False)

    def SaveModel(self):
        """
        Save the classification models in the data directory
        """
        print "Saving the models"
        parameters = [self.__editus_param_matrix]
        for parameter in parameters:
            param_name = parameter.split('/')[-1:][0][:-4]
            y_digits = numpy.loadtxt(self.__gender_path, dtype=float)
            X_digits = numpy.loadtxt(parameter, dtype=float)
            X_train = X_digits[:3000]
            y_train = y_digits[:3000]

            ###### same male and female count ########
            # X_male = []
            # y_male = []
            # X_female = []
            # y_female = []
            # for p,g in zip(X_train, y_train):
            #     if g == 0:
            #         X_female.append(p)
            #         y_female.append(0)
            #     elif g == 1:
            #         X_male.append(p)
            #         y_male.append(1)
            # X_male = numpy.asarray(X_male)
            # X_female = numpy.asarray(X_female)
            # # X_male = numpy.vstack(set(map(tuple, X_male))) # get unique entries
            # X_male = X_male[~numpy.all(X_male == 0, axis=1)] # remove zero entries in male training
            # X_female = X_female[~numpy.all(X_female == 0, axis=1)] # remove zero entries in female training
            # f_count = len(X_female)
            # # X_male = X_male[:f_count]
            # # y_male = y_male[:f_count]
            # X_train = numpy.concatenate((X_male,X_female),axis=0)
            # y_train = numpy.asarray(y_male + y_female)
            #############################################
            number = 600
            gradient_boost = ensemble.GradientBoostingClassifier(n_estimators=number, learning_rate=0.05, max_depth=10)
            extra_trees = ensemble.ExtraTreesClassifier(n_estimators=number, n_jobs=-1)
            bagging = ensemble.BaggingClassifier(n_estimators=number, n_jobs=-1)
            svm_ = svm.SVC(kernel='linear', probability=True)
            gaussian = GaussianProcessClassifier(n_jobs=-1, max_iter_predict=1000)
            classifiers = [gradient_boost, extra_trees, bagging, svm_, gaussian]
            for classifier in classifiers:
                c_name = self.__model_path + str(param_name) + '_' + str(classifier).split('(')[0] + '.pkl'
                classifier.fit(X_train, y_train)
                joblib.dump(classifier, c_name)
                print classifier
                print


def main():
    folder_path = os.path.realpath(__file__)[:-17]
    data_directory = folder_path + "data/"
    model_directory = data_directory + "model/"
    to_delete = [f for f in os.listdir(data_directory) if f.endswith(".dat")]
    for f in to_delete:
        if "parameters" in f or "all_cat" in f:
            pass
        else:
            os.remove(data_directory + f)
    gender_prediction = GenderPrediction_rtlcategories(data_path=data_directory, total_number_of_users=None,
                                                       model_directory=model_directory)
    gender_prediction.GetAllData2()
    gender_prediction.ConstructParameterMatrix()
    gender_prediction.SaveModel()


if __name__ == "__main__":
    main()
