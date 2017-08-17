#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ML, io, numpy, string, os
from pymongo import MongoClient

corpus_file = "/home/lt/PycharmProjects/Recommendation/data/bleilab-python/5000_corpus.txt"
test_corpus = "/home/lt/PycharmProjects/Recommendation/data/bleilab-python/test_corpus.txt"
vocab_file = "/home/lt/PycharmProjects/Recommendation/data/bleilab-python/5000_vocab.txt"
_gamma_location = "/home/lt/PycharmProjects/Recommendation/data/bleilab-python/gamma-1000.dat"
_lambda_location = "/home/lt/PycharmProjects/Recommendation/data/bleilab-python/lambda-1000.dat"

# calculate perplexity
ML.calculate_perplexity(corpus=test_corpus, vocab=vocab_file, _lambda=_lambda_location, _gamma=_gamma_location,
                        _alpha=0.1, _beta=0.01)

# save the output of nmf in webpage
ML.visualize_lda(corpus=corpus_file, doc_topic_mat=_gamma_location, topic_word_mat=_lambda_location,
                 vocabulary_file=vocab_file, output_file="/home/lt/Downloads/blei_lda_output.html")
