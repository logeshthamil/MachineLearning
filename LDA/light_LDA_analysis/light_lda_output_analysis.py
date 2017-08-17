#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ML, io, numpy, string, os
from pymongo import MongoClient

corpus_file = "/home/lt/PycharmProjects/Recommendation/data/lightlda/5000_corpus.txt"
test_corpus = "/home/lt/PycharmProjects/Recommendation/data/lightlda/test_corpus.txt"
vocab_file = "/home/lt/PycharmProjects/Recommendation/data/lightlda/LightLDA-input/QX_RTLDB_5000.vocab.txt"
doc_topic_output_file = "/home/lt/PycharmProjects/Recommendation/data/lightlda/LightLDA-output/doc_topic.0"
topic_word_output_file = "/home/lt/PycharmProjects/Recommendation/data/lightlda/LightLDA-output/server_0_table_0.model"
total_topic_output_file = "/home/lt/PycharmProjects/Recommendation/data/lightlda/LightLDA-output/server_0_table_1.model"
_gamma_location = "/home/lt/PycharmProjects/Recommendation/data/lightlda/gamma.txt"
_lambda_location = "/home/lt/PycharmProjects/Recommendation/data/lightlda/lambda.txt"

directory = "/home/lt/PycharmProjects/Recommendation/data/lightlda/"
test_count = 500
# the articles
n_samples = None
n_features = None
n_topics = 25
n_top_words = 20

##########################################################
# client = MongoClient('localhost', 27017)
uri = 'mongodb://qxuser:userqx16@52.57.103.28/?authSource=users'
client = MongoClient(uri)
collection = client.RtlDB.Articles
test_data = collection.find({}, {"url": 0, "title": 0, "kicker": 0, "header": 0, "tags": 0}). \
    sort([['_id', -1]]).limit(test_count)  # 1 for sorting in ascending order and -1 for sorting in descending order


def check_vocab(input_word):
    """
    return False if special char (defined) are found in input_word
    return True otherwise
    """
    # ... replace("char you allow","char you don t allow")
    invalid_chars = set(string.punctuation.replace("", "’ -'´`"))
    if any(char in invalid_chars for char in input_word):
        return False
    elif len(input_word) < 5:
        return False
    else:
        return True


def get_test_data():
    """
    Get the test data from the mongo db server and save it in desired format.
    """
    print "Getting test data from the mongo db server"
    with io.open(test_corpus, 'w', encoding='utf-8') as file1:
        n = 1
        for datas_all in test_data:
            datas = datas_all.get('tf')
            # check if the tf vector is empty
            if not datas:
                # print "Empty entry:" + str(datas_all)
                pass
            else:
                # file1.write(str(n).decode("utf-8") + u' en ') # in case of gensim library include this line
                for data in datas:
                    vocab = numpy.array(data.keys()[0])
                    # update vocab file
                    if check_vocab(data.keys()[0]):
                        # write the data to the corpus file
                        for i in range(int(data.values()[0])):
                            file1.write(data.keys()[0] + ' ')
                file1.write(u"\n")
                n = n + 1
    file1.close()


# get test data for perplexity calculation
if not os.path.isfile(test_corpus):
    get_test_data()

# transform the output of the lda to readable and normalized format and save it in a file
ML.transform_lightlda_outputs(doc_topic=doc_topic_output_file, topic_word=topic_word_output_file,
                              total_topic=total_topic_output_file, save=True, doc_topic_location=_gamma_location,
                              topic_word_location=_lambda_location)

# calculate perplexity
ML.calculate_perplexity(corpus=test_corpus, vocab=vocab_file, _lambda=_lambda_location, _gamma=_gamma_location,
                        _alpha=0.1, _beta=0.01)

# save the output of nmf in webpage
ML.visualize_lda(corpus=corpus_file, doc_topic_mat=_gamma_location, topic_word_mat=_lambda_location,
                 vocabulary_file=vocab_file, output_file="/home/lt/Downloads/lightlda_output_tsne.html")
