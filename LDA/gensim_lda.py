# !/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim import corpora
from gensim import models
import time, os, numpy, string, io
from pymongo import MongoClient
import pyLDAvis.gensim

directory = "/home/lt/PycharmProjects/Recommendation/data/gensim/"
count_db = 5000  # get only the first <count_db> elements of the mongodb articles, if <count_db> is null it will get all
test_count = 1000
# the articles

##########################################################
# client = MongoClient('localhost', 27017)
uri = 'mongodb://qxuser:userqx16@52.57.103.28/?authSource=users'
client = MongoClient(uri)
collection = client.RtlDB.Articles
if count_db is None:
    count_db = collection.find().count()
all_data = collection.find({}, {"url": 0, "title": 0, "kicker": 0, "header": 0, "tags": 0}). \
    sort([['_id', 1]]).limit(count_db)  # 1 for sorting in ascending order and -1 for sorting in descending order
test_data = collection.find({}, {"url": 0, "title": 0, "kicker": 0, "header": 0, "tags": 0}). \
    sort([['_id', -1]]).limit(test_count)  # 1 for sorting in ascending order and -1 for sorting in descending order
corpus_location = directory + str(count_db) + "_corpus.txt"
article_id = directory + str(count_db) + "_article_id.txt"
vocabulary = directory + str(count_db) + "_vocab.txt"
test_corpus = directory + "test_corpus.txt"


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


def get_data():
    """
    Get the data from the mongo db server and save it in desired format.
    """
    print "Getting data from the mongo db server"
    print "Total number of articles: " + str(all_data.count())
    vocabs = []
    article_doc = io.open(article_id, 'w', encoding='utf-8')
    with io.open(corpus_location, 'w', encoding='utf-8') as file1:
        n = 1
        for datas_all in all_data:
            datas = datas_all.get('tf')
            # check if the tf vector is empty
            if not datas:
                # print "Empty entry:" + str(datas_all)
                pass
            else:
                article_doc.write(datas_all.get('id') + '\n')
                file1.write(str(n).decode("utf-8") + u' en ')  # in case of gensim library include this line
                for data in datas:
                    vocab = numpy.array(data.keys()[0])
                    # update vocab file
                    if check_vocab(data.keys()[0]):
                        if not numpy.any(vocabs[:] == vocab):
                            vocabs = numpy.append(vocabs, vocab)
                        # write the data to the corpus file
                        for i in range(int(data.values()[0])):
                            file1.write(data.keys()[0] + ' ')
                file1.write(u"\n")
                n = n + 1
    # write the vocabulary in the vocab file
    with io.open(vocabulary, 'w', encoding='utf-8') as v:
        for vocab in vocabs:
            v.write(vocab + '\n')
    file1.close()
    v.close()


def get_test_data():
    """
    Get the test data from the mongo db server and save it in desired format.
    """
    print "Getting test data from the mongo db server"
    print "Total number of test articles: " + str(all_data.count())
    with io.open(test_corpus, 'w', encoding='utf-8') as file1:
        n = 1
        for datas_all in test_data:
            datas = datas_all.get('tf')
            # check if the tf vector is empty
            if not datas:
                # print "Empty entry:" + str(datas_all)
                pass
            else:
                file1.write(str(n).decode("utf-8") + u' en ')  # in case of gensim library include this line
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


if not os.path.isfile(corpus_location):
    get_data()
if not os.path.isfile(test_corpus):
    get_test_data()
corpus = corpora.MalletCorpus(corpus_location)
test_corpus = corpora.MalletCorpus(test_corpus)
time_iterations = []
for iterations in range(500, 1000, 50):
    time1 = time.time()
    lda_model = models.LdaMulticore(corpus=corpus, num_topics=25, iterations=iterations, alpha=0.1, eta=0.01,
                                    id2word=corpus.id2word,
                                    chunksize=1000)
    # lsi_model = models.LsiModel(corpus=corpus, num_topics=25, id2word=corpus.id2word)
    time2 = time.time()
    print "number of iterations: " + str(iterations)
    print "time taken to compute lda on the data: %r seconds" % str(time2 - time1)
    print "per word bound: " + str(lda_model.log_perplexity(chunk=test_corpus))
    print "perplexity: " + str(2 ** (-lda_model.log_perplexity(chunk=test_corpus)))
    print

# for topic in lda_model.print_topics(num_topics=100, num_words=20):
#     id = topic[0]
#     elem = topic[1]
#     print "topic" + str(id)
#     for e in elem.split('+'):
#         print e
#     print
