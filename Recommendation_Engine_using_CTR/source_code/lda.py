import os, io, time, numpy, shutil, sklearn.decomposition, helper_functions, subprocess
from sklearn.feature_extraction.text import CountVectorizer


class LatentDirichletAllocation(object):
    """
    An abstract base class for all Latent Dirichlet Allocation algorithms.
    """

    def __init__(self, number_of_topics=25, iterations=1000, alpha=0.1, beta=0.01, data_path=None):
        """
        :param number_of_topics: the total number of topics which has to be extracted
        :param iterations: the total number of iterations for the lda
        :param alpha: the alpha value
        :param beta: the beta value
        :param data_path: the path in which the data can be saved
        """
        self.__data_path = data_path
        self._corpus_location = os.path.join(self.__data_path, "corpus.dat")
        self._iterations = iterations
        self._alpha = alpha
        self._beta = beta
        self._number_of_topics = number_of_topics
        self._gamma_location = os.path.join(self.__data_path, "gamma.dat")
        self._lambda_location = os.path.join(self.__data_path, "lambda.dat")
        self._vocab_location = os.path.join(self.__data_path, "lda_vocabulary.dat")
        self._vocab_full_location = os.path.join(self.__data_path, "vocabulary.dat")
        self._mult_location = os.path.join(self.__data_path, "mult.dat")

    def _ctr_data_generate(self):
        """
        A method to generate the input corpus data to ctr from the standard corpus data and save it with file name
        mult.dat.
        """
        print "Transforming corpus to lib svm format for ctr"
        corpus = io.open(self._corpus_location, mode='r').readlines()
        vocab = io.open(self._vocab_location, mode='r').readlines()
        vocab = map(lambda s: s.strip(), vocab)
        mult = io.open(self._mult_location, mode='w', encoding=None)
        for line in corpus:
            line_vocabs = line.split(" ")[:-1]
            index = {}
            for line_vocab in line_vocabs:
                try:
                    i = vocab.index(line_vocab)
                    if i in index.keys():
                        index[i] += 1
                    else:
                        index.update({i: 1})
                except:
                    pass
            length_of_line = str(len(index)) + ' '
            mult.write(length_of_line.decode('unicode-escape'))
            for i in index:
                temp = str(i) + ":" + str(index[i]) + " "
                mult.write(temp.decode('unicode-escape'))
            mult.write(u'\n')

    def _transform_data(self):
        """
        This method should be overridden by the derived classes.
        """
        raise NotImplementedError("This method should be overwritten by the derived classes")

    def GetOutput(self):
        """
        This method should be overridden by the derived classes.
        """
        raise NotImplementedError("This method should be overwritten by the derived classes")


class SklearnLDA(LatentDirichletAllocation):
    """
    A derived class of LatentDirichletAllocation base class which applies the LDA technique of sklearn library.
    """

    def __init__(self, number_of_topics=25, iterations=1000, alpha=0.1, beta=0.01, data_path=None):
        """
        :param number_of_topics: the total number of topics which has to be extracted
        :param iterations: the total number of iterations of the LDA
        :param alpha: the alpha value
        :param beta: the beta value
        :param data_path: the path in which the data is stored
        """
        LatentDirichletAllocation.__init__(self, number_of_topics=number_of_topics,
                                           iterations=iterations, alpha=alpha, beta=beta, data_path=data_path)
        self._transform_data()

    def _transform_data(self):
        """
        A method to transform the corpus data to the format which is suitable for the LDA algorithm.
        """
        self.__transformed_corpus = self._corpus_location

    def GetOutput(self):
        """
        A method to apply LDA on the transformed data and save the output of LDA in corresponding locations.
        """
        print "Applying LDA from sklearn library on the corpus data" + '\n'
        data_samples = io.open(self.__transformed_corpus, encoding='utf-8').readlines()
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2)
        tf = tf_vectorizer.fit_transform(data_samples)
        vocab = sorted(tf_vectorizer.vocabulary_)
        lda = sklearn.decomposition.LatentDirichletAllocation(n_topics=self._number_of_topics,
                                                              max_iter=self._iterations,
                                                              learning_method='online',
                                                              learning_offset=50.,
                                                              random_state=0, batch_size=1000, n_jobs=1,
                                                              doc_topic_prior=self._alpha,
                                                              topic_word_prior=self._beta)
        t0 = time.time()
        W = lda.fit_transform(tf)
        H = lda.components_
        print("Sklearn LDA: time taken for computation: %0.3f seconds." % (time.time() - t0))
        print
        # save the model
        numpy.savetxt(self._gamma_location, W)
        numpy.savetxt(self._lambda_location, H)
        with io.open(self._vocab_location, mode='w') as f:
            for v in vocab:
                f.write(v)
                f.write(u'\n')
        print "Sklearn LDA model saved"
        self._ctr_data_generate()


class LightLDA(LatentDirichletAllocation):
    """
    A derived class of LatentDirichletAllocation base class which acts as a wrapper to use the LightLDA from DMTK.
    """

    def __init__(self, number_of_topics=25, iterations=1000, alpha=0.1, beta=0.01, data_path=None):
        """
        :param number_of_topics: the total number of topics which has to be extracted
        :param iterations: the total number of iterations of the LDA
        :param alpha: the alpha value
        :param beta: the beta value
        :param data_path: the path in which the data can be stored
        """
        LatentDirichletAllocation.__init__(self, number_of_topics=number_of_topics, iterations=iterations, alpha=alpha,
                                           beta=beta, data_path=data_path)
        self.__data_path = data_path
        self.__transformed_corpus = os.path.join(self.__data_path, "lightlda_corpus.dat")
        self.__transformed_vocab = os.path.join(self.__data_path, "lightlda_vocab.dat")
        self._transform_data()

    def _transform_data(self):
        """
        A method to transform the corpus from standard format to libsvm format to use it as an input for lightLDA.
        """
        # generating the input corpus file for light lda
        print "converting the standard corpus to libsvm format to use it as input for light lda"
        vocab_location = io.open(self._vocab_full_location, mode='r').readlines()
        vocab = map(lambda s: s.strip(), vocab_location)
        self._len_vocab = len(vocab)
        corpus_location = io.open(self._corpus_location, mode='r').readlines()
        corpus = map(lambda s: s.strip(), corpus_location)
        self._len_corpus = len(corpus)
        transformed_corpus = io.open(self.__transformed_corpus, mode='w')
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
        transformed_vocab = io.open(self.__transformed_vocab, mode='w')
        count = {}
        for w in io.open(self._corpus_location).read().split():
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

    def GetOutput(self):
        """
        A method to apply LDA on the transformed data and save the output of LDA in corresponding locations.
        """
        corpus = self.__transformed_corpus
        vocab = self.__transformed_vocab
        print "Converting the input data of light lda to binary" + "\n"
        script_path = os.path.abspath(__file__)
        # lda_path = os.path.join(os.path.split(script_path)[0][:-11], "source_code/lightlda/bin/")
        lda_path = "/usr/local/bin/"
        data_path = os.path.join(self.__data_path, "lda_outputs/")
        os.chdir(lda_path)
        transform_to_binary_command = "./dump_binary " + corpus + " " + vocab + " " + data_path + " 0"
        os.system(transform_to_binary_command)

        print "Applying lda on binary data" + "\n"
        cmd = ['./lightlda', '-num_vocabs', str(self._len_vocab + 200), '-num_topics', str(self._number_of_topics),
               '-num_iterations', str(self._iterations), '-mh_steps', '2', '-alpha', str(self._alpha), '-beta',
               str(self._beta), '-num_blocks', '1', '-max_num_document', str(self._len_corpus + 200),
               '-input_dir', data_path[:-1], '-num_servers', '1', 'num_local_workers', '7', '-data_capacity', '800']
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for line in p.stdout:
            pass
        p.wait()
        print("LightLDA: time taken for computation: %0.3f seconds." % (time.time() - t0))

        # copy the output of lda from the bin directory to data directory
        file1 = lda_path + "server_0_table_0.model"
        file2 = lda_path + "server_0_table_1.model"
        file3 = lda_path + "doc_topic.0"
        n_file1 = data_path + "server_0_table_0.model"
        n_file2 = data_path + "server_0_table_1.model"
        n_file3 = data_path + "doc_topic.0"
        shutil.copyfile(file1, n_file1)
        shutil.copyfile(file2, n_file2)
        shutil.copyfile(file3, n_file3)

        # extract the gamma and lambda matrix from the output of lda and save it the output directory
        print "Extracting the lambda and gamma matrix from the output of the light lda" + '\n'
        helper_functions.transform_lightlda_outputs(doc_topic=n_file3, topic_word=n_file1, total_topic=n_file2,
                                                    doc_topic_location=self._gamma_location,
                                                    topic_word_location=self._lambda_location, save=True)
        self._vocab_location = self._vocab_full_location
        self._ctr_data_generate()


class LightLDA_for_recommendation(LatentDirichletAllocation):
    """
    A derived class of LatentDirichletAllocation base class which acts as a wrapper to use the LightLDA from DMTK.
    """

    def __init__(self, number_of_topics=25, iterations=1000, alpha=0.1, beta=0.01, data_path=None):
        """
        :param number_of_topics: the total number of topics which has to be extracted
        :param iterations: the total number of iterations of the LDA
        :param alpha: the alpha value
        :param beta: the beta value
        :param data_path: the path in which the data can be stored
        """
        LatentDirichletAllocation.__init__(self, number_of_topics=number_of_topics, iterations=iterations, alpha=alpha,
                                           beta=beta, data_path=data_path)
        self.__data_path = data_path
        self.__transformed_corpus = os.path.join(self.__data_path, "lda_lightlda_corpus.dat")
        self.__transformed_vocab = os.path.join(self.__data_path, "lda_lightlda_vocab.dat")
        self._gamma_location = os.path.join(self.__data_path, "lda_gamma.dat")
        self._lambda_location = os.path.join(self.__data_path, "lda_lambda.dat")
        self._vocab_full_location = os.path.join(self.__data_path, "lda_item_id.dat")
        self._mult_location = os.path.join(self.__data_path, "lda_mult.dat")
        self._corpus_location = os.path.join(self.__data_path, "lda_corpus.dat")
        self._transform_data()

    def _transform_data(self):
        """
        A method to transform the corpus from standard format to libsvm format to use it as an input for lightLDA.
        """
        # generating the input corpus file for light lda
        print "converting the standard corpus to libsvm format to use it as input for light lda"
        vocab_location = io.open(self._vocab_full_location, mode='r').readlines()
        vocab = map(lambda s: s.strip(), vocab_location)
        self._len_vocab = len(vocab)
        corpus_location = io.open(self._corpus_location, mode='r').readlines()
        corpus = map(lambda s: s.strip(), corpus_location)
        self._len_corpus = len(corpus)
        transformed_corpus = io.open(self.__transformed_corpus, mode='w')
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
        transformed_vocab = io.open(self.__transformed_vocab, mode='w')
        count = {}
        for w in io.open(self._corpus_location).read().split():
            if w in count:
                count[w] += 1
            else:
                count[w] = 1
        for i, c in enumerate(vocab):
            try:
                t1 = str(i) + '\t'
                t2 = c
                t3 = '\t' + str(count[c]) + '\n'
                transformed_vocab.write(t1.decode('unicode-escape'))
                transformed_vocab.write(t2)
                transformed_vocab.write(t3.decode('unicode-escape'))
            except:
                pass

    def GetOutput(self):
        """
        A method to apply LDA on the transformed data and save the output of LDA in corresponding locations.
        """
        corpus = self.__transformed_corpus
        vocab = self.__transformed_vocab
        print "Converting the input data of light lda to binary" + "\n"
        script_path = os.path.abspath(__file__)
        lda_path = os.path.join(os.path.split(script_path)[0][:-11], "source_code/lightlda/bin/")
        data_path = os.path.join(self.__data_path, "lda_outputs/")
        os.chdir(lda_path)
        transform_to_binary_command = "./dump_binary " + corpus + " " + vocab + " " + data_path + " 0"
        os.system(transform_to_binary_command)
        os.chdir(lda_path)
        print "Applying lda on binary data" + "\n"
        cmd = ['./lightlda', '-num_vocabs', str(self._len_vocab + 200), '-num_topics', str(self._number_of_topics),
               '-num_iterations', str(self._iterations), '-mh_steps', '2', '-alpha', str(self._alpha), '-beta',
               str(self._beta), '-num_blocks', '1', '-max_num_document', str(self._len_corpus + 200),
               '-input_dir', data_path[:-1], '-num_servers', '1', 'num_local_workers', '7', '-data_capacity', '800']
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for line in p.stdout:
            pass
        p.wait()
        print("LightLDA: time taken for computation: %0.3f seconds." % (time.time() - t0))

        # copy the output of lda from the bin directory to data directory
        file1 = lda_path + "server_0_table_0.model"
        file2 = lda_path + "server_0_table_1.model"
        file3 = lda_path + "doc_topic.0"
        n_file1 = data_path + "server_0_table_0.model"
        n_file2 = data_path + "server_0_table_1.model"
        n_file3 = data_path + "doc_topic.0"
        shutil.copyfile(file1, n_file1)
        shutil.copyfile(file2, n_file2)
        shutil.copyfile(file3, n_file3)

        # extract the gamma and lambda matrix from the output of lda and save it the output directory
        print "Extracting the lambda and gamma matrix from the output of the light lda" + '\n'
        helper_functions.transform_lightlda_outputs(doc_topic=n_file3, topic_word=n_file1, total_topic=n_file2,
                                                    doc_topic_location=self._gamma_location,
                                                    topic_word_location=self._lambda_location, save=True)
        self._vocab_location = self._vocab_full_location
        self._ctr_data_generate()

        # copy the output of the lda to the ctr folder to get recommendation
        v = self.__data_path + "ctr_outputs/final-V.dat"
        u = self.__data_path + "ctr_outputs/final-U.dat"
        lamb = numpy.loadtxt(self._lambda_location)
        shutil.copyfile(self._gamma_location, u)
        numpy.savetxt(v, numpy.transpose(lamb))
