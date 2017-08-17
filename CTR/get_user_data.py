#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pymongo import MongoClient
import io, numpy

directory = "/home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data/"
count_users = 50000  # get only the first <count_users> elements of the mongodb users database,
# if <count_users> is null it will get all users
user_article_file = "users.dat"
user_id_file = "user_id.dat"
article_user_file = "items.dat"
corpus_location = "/home/lt/PycharmProjects/Recommendation/data/lightlda/LightLDA-input/QX_RTLDB_5000.corpus.libsvm"
article_id_file = "/home/lt/PycharmProjects/Recommendation/data/lightlda/LightLDA-input/QX_RTLDB_5000.article_id.txt"
new_corpus_location = "/home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data/corpus.dat"

##########################################################
# client = MongoClient('localhost', 27017)
uri = 'mongodb://qxuser:userqx16@52.57.103.28/?authSource=users'
client = MongoClient(uri)
collection = client.RtlDB.Users
if count_users is None:
    count_users = collection.find().count()
all_users = collection.find().limit(count_users)
user_directory = directory + user_article_file
user_id_directory = directory + user_id_file
article_user_directory = directory + article_user_file

article_ids = io.open(article_id_file, mode='r').readlines()
article_ids = map(lambda s: s.strip(), article_ids)


def get_user_article_file():
    """
    A function to generate the users.dat file and userid.dat file for recommendation using collaborative topic
    regression.
    """
    user_id_file = io.open(user_id_directory, mode='w')
    with io.open(user_directory, mode='w') as user_file:
        for user in all_users:
            user_articles = user.get('articles')
            user_id = user.get("id")
            valid_articles = []
            for user_article in user_articles:
                id = user_article.get('id')
                if id in article_ids:
                    valid_articles.append(id)
            # condition to check the number of valid articles, reduces noise
            if len(valid_articles) > 2:
                user_id_file.write(user_id)
                length = str(len(valid_articles)).decode('utf-8') + u' '
                user_file.write(length)
                for article in valid_articles:
                    article = str(article_ids.index(article)).decode('unicode-escape')
                    a_id = article + u' '
                    user_file.write(a_id)
                user_file.write(u'\n')
                user_id_file.write(u'\n')


def change_svmcorpusfile_to_ctrcorpus():
    """
    Get the data from the input file of the light LDA and transform into the input file of the collaborative topic
    regression.
    """
    nc = io.open(new_corpus_location, encoding='utf8', mode='w')
    with io.open(corpus_location, encoding='utf8') as corpus:
        corpus_lines = corpus.readlines()
        for corpus_line in corpus_lines:
            length = len(corpus_line.split('\t')[1:][0].split(' ')[:-1])
            nc.write(str(length).decode('utf-8'))
            nc.write(u' ')
            nc.write(corpus_line.split('\t')[1:][0])


# some error in this function, need to debug
def get_article_user_file():
    """
    A function to generate the articles.dat from the article_id file for recommendation using collaborative topic
    regression.
    """
    print "get article user file"
    user_id_values = io.open(user_id_directory, mode='r').readlines()
    user_id_values = map(lambda s: s.strip(), user_id_values)
    len_articleids = len(article_ids)
    temp_article_user = ["", ] * len_articleids
    items = io.open(article_user_directory, mode='w')
    for user in all_users:
        user_articles = user.get('articles')
        user_id = user.get("id")
        for user_article in user_articles:
            a_id = user_article.get('id')
            print a_id
            if a_id in article_ids:
                try:
                    index = user_id_values.index(user_id)
                    # user_id_index = user_id_values.index(user_id)
                    temp_article_user[index] = temp_article_user[index] + str(index) + " "
                except:
                    pass
    for item in temp_article_user:
        total_items = len(item.split(" "))
        items.write(u"%d %s\n" % (total_items - 1, item))


def get_article_user_file1():
    """
    A function to generate the articles.dat from the article_id file for recommendation using collaborative topic
    regression.
    """
    articles_id = io.open(article_id_file, mode='r').readlines()
    articles_id = map(lambda s: s.strip(), articles_id)
    temp_articles_users = ["", ] * len(articles_id)
    users_article = numpy.loadtxt(user_directory, dtype=str, delimiter="\n")
    for user_id, user_article in enumerate(users_article):
        user_article = user_article.split(" ")[1:-1]
        for i_user_article in user_article:
            temp_articles_users[int(i_user_article)] = str(temp_articles_users[int(i_user_article)]) + str(
                user_id) + " "
    aud = io.open(article_user_directory, mode='w')
    for temp_article_user in temp_articles_users:
        aud.write(str(len(temp_article_user.split())).decode('unicode-escape'))
        aud.write(u' ')
        aud.write(temp_article_user.decode('unicode-escape'))
        aud.write(u'\n')


get_user_article_file()
change_svmcorpusfile_to_ctrcorpus()
get_article_user_file1()
