#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pymongo import MongoClient
import os
import numpy
import string
from time import time
import ML.lda_visualize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
import io

directory = "/home/lt/PycharmProjects/Recommendation/data/sklearn_nmf/"
count_db = 5000  # get only the first <count_db> elements of the mongodb articles, if <count_db> is null it will get all
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
if count_db is None:
    count_db = collection.find().count()
all_data = collection.find({}, {"url": 0, "title": 0, "kicker": 0, "header": 0, "tags": 0}). \
    sort([['_id', 1]]).limit(count_db)  # 1 for sorting in ascending order and -1 for sorting in descending order
test_data = collection.find({}, {"url": 0, "title": 0, "kicker": 0, "header": 0, "tags": 0}). \
    sort([['_id', -1]]).limit(test_count)  # 1 for sorting in ascending order and -1 for sorting in descending order
corpus_location = directory + str(count_db) + "_corpus.txt"
article_id = directory + str(count_db) + "_article_id.txt"
vocabulary_file = directory + str(count_db) + "_vocab.txt"
test_corpus = directory + "test_corpus.txt"
_gamma = directory + "gamma.txt"
_lambda = directory + "lambda.txt"
vocab_path = directory + "vocab_used.txt"


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
                # file1.write(str(n).decode("utf-8") + u' en ') # in case of gensim library include this line
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
    with io.open(vocabulary_file, 'w', encoding='utf-8') as v:
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


if not os.path.isfile(corpus_location):
    get_data()
if not os.path.isfile(test_corpus):
    get_test_data()

vocabulary = io.open(vocabulary_file, encoding='utf-8').readlines()


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print("\n".join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print
    print


print "Loading rtl dataset..."
t0 = time()
data_samples = io.open(corpus_location, encoding='utf-8').readlines()
print "done in %0.3fs." % (time() - t0)

# Use tf-idf features for NMF.
print "Extracting tf-idf features for NMF..."
# tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
t0 = time()
tfidf = tfidf_vectorizer.fit_transform(data_samples)
vocab = sorted(tfidf_vectorizer.vocabulary_)
print "done in %0.3fs." % (time() - t0)

# Fit the NMF model
print "Fitting the NMF model with tf-idf features, ""n_samples=%r and n_features=%r..." % (n_samples, n_features)
t0 = time()
nmf_model = NMF(n_components=n_topics, random_state=1, alpha=0.1, l1_ratio=.5, beta=0.01, max_iter=1000)
W = nmf_model.fit_transform(tfidf)
H = nmf_model.components_
print "done in %0.3fs." % (time() - t0)

# print type(W)
# print H.shape

# save the model
numpy.savetxt(_gamma, W)
numpy.savetxt(_lambda, H)
with io.open(vocab_path, 'w', encoding='utf8') as v:
    for v_i in vocab:
        v.write(v_i + '\n')

# calculate perplexity
from ML.perplexity import calculate_perplexity

calculate_perplexity(corpus=test_corpus, vocab=vocab_path, _lambda=_lambda, _gamma=_gamma, _alpha=0.1, _beta=0.01)

print "\nTopics in NMF model:"
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
print_top_words(nmf_model, tfidf_feature_names, n_top_words)

# save the output of nmf in webpage
ML.lda_visualize.visualize_lda(corpus=corpus_location, doc_topic_mat=_gamma, topic_word_mat=_lambda,
                               vocabulary_file=vocab_path,
                               output_file="/home/lt/Downloads/sk_learn_nmf_output.html")
