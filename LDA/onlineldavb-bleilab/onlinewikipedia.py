# !/usr/bin/env python
# -*- coding: utf-8 -*-

# onlinewikipedia.py: Demonstrates the use of online VB for LDA to
# analyze a bunch of random Wikipedia articles.
#
# Copyright (C) 2010  Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import onlineldavb
import os, numpy, string, io
from pymongo import MongoClient

directory = "/home/lt/PycharmProjects/Recommendation/data/bleilab-python/"
count_db = 5000  # get only the first <count_db> elements of the mongodb articles, if <count_db> is null it will get all
test_count = 500
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
vocabulary_path = directory + str(count_db) + "_vocab.txt"
test_corpus = directory + "test_corpus.txt"
_lambda = directory + "lambda.txt"
_gamma = directory + "gamma.txt"


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
                # file1.write(str(n).decode("utf-8") + u' en ')  # in case of gensim library include this line
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
    with io.open(vocabulary_path, 'w', encoding='utf-8') as v:
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
                # file1.write(str(n).decode("utf-8") + u' en ')  # in case of gensim library include this line
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


def main():
    """
    Downloads and analyzes a bunch of random Wikipedia articles using
    online VB for LDA.
    """
    if not os.path.isfile(corpus_location):
        get_data()
    if not os.path.isfile(test_corpus):
        get_test_data()

    vocab = file(vocabulary_path).readlines()
    docset = file(corpus_location).readlines()
    articlenames = file(article_id).readlines()

    # The number of documents to analyze each iteration
    batchsize = 1000
    # The total number of documents in Wikipedia
    D = len(articlenames)
    # The number of topics
    K = 25
    alpha = 0.1
    beta = 0.01

    # How many documents to look at
    if (len(sys.argv) < 2):
        documentstoanalyze = int(D / batchsize)
    else:
        documentstoanalyze = int(sys.argv[1])

    # Our vocabulary
    W = len(vocab)
    documentstoanalyze = 1001

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, alpha, beta, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    for iteration in range(0, documentstoanalyze):
        print iteration
        # Download some articles
        # (docset, articlenames) = \
        #     wikirandom.get_random_wikipedia_articles(batchsize)
        # Give them to online LDA
        (gamma, bound) = olda.update_lambda_docs(docset)
        # Compute an estimate of held-out perplexity
        # (wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        # perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        # print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
        #     (iteration, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 1 == 0):
            numpy.savetxt(_lambda, olda._lambda)
            numpy.savetxt(_gamma, gamma)


if __name__ == '__main__':
    main()
