# -*- coding: utf-8 -*-
from pymongo import MongoClient
import os, io, numpy, string, warnings, shutil


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


def modify_user_item(user_item_directory, delete_items=1, valid_users=3, test_file=None, user_item_copy_file=None,
                     test_recommender='in'):
    """
    A function to modify the user item file. This function deletes certain items to detect it using recommendation
    engine. This function deletes in two modes. In the in mode this function deletes the items inside the matrix.
    Whereas in the out mode this function deletes the entire user visits.
    :param user_item_directory: the file in which the user item matrix is stored
    :param delete_items: the number of items which has to be deleted
    :param valid_users: the number of items which has to be visited by a valid user
    :param test_recommender: either 'in' or 'out'
    """
    if test_recommender == 'in':
        print "Modifying user item file for in matrix evaluation"
        shutil.copyfile(user_item_directory, user_item_copy_file)
        users_item = numpy.loadtxt(user_item_directory, dtype=str, delimiter="\n")
        users_item_new = io.open(user_item_directory, mode='w')
        user_item_test_file = io.open(test_file, mode='w')
        temp = []
        for user_id, user_item__ in enumerate(users_item):
            user_items = user_item__.split(" ")[1:-1]
            indexes = numpy.unique(user_items, return_index=True)[1]
            unique_items = [user_items[index] for index in sorted(indexes)]
            if len(unique_items) > valid_users:
                items_to_delete = unique_items[-delete_items:]
                for item_to_delete in items_to_delete:
                    user_items = filter(lambda a: a != item_to_delete, user_items)
                temp.append(user_items)
                t = str(user_id) + ' ' + ' '.join(items_to_delete) + '\n'
                user_item_test_file.write(t.decode('unicode-escape'))
            else:
                temp.append(user_items)
        for t in temp:
            users_item_new.write(str(len(t)).decode('unicode-escape'))
            users_item_new.write(u' ')
            users_item_new.write(' '.join(t).decode('unicode-escape'))
            users_item_new.write(u'\n')
    elif test_recommender == 'out':
        print "Modifying user item file for out matrix evaluation"
        shutil.copyfile(user_item_directory, user_item_copy_file)
        users_item = numpy.loadtxt(user_item_directory, dtype=str, delimiter="\n")
        users_item_new = io.open(user_item_directory, mode='wb')
        user_item_test_file = io.open(test_file, mode='wb')
        items_to_delete = []
        # find the articles which has to be deleted
        for user_id, user_item__ in enumerate(users_item):
            if len(items_to_delete) >= delete_items:
                break
            else:
                user_items = user_item__.split(" ")[1:-1]
                indexes = numpy.unique(user_items, return_index=True)[1]
                unique_items = [user_items[index] for index in sorted(indexes)]
                if len(unique_items) > valid_users:
                    items_to_delete.append(unique_items[-1])

        # delete the found articles from all the users
        def filter_list(full_list, excludes):
            s = set(excludes)
            return (x for x in full_list if x not in s)

        a = 0
        item_user_deletions = [[] for i in range(len(items_to_delete))]
        for user_id, user_item__ in enumerate(users_item):
            user_items = user_item__.split(" ")[1:-1]
            filtered_list = list(filter_list(user_items, items_to_delete))
            if len(filtered_list) != len(user_items):
                a = a + 1
                deleted_items = [x for x in user_items if x not in filtered_list]
                indexes = numpy.unique(deleted_items, return_index=True)[1]
                deleted_items_u = [deleted_items[index] for index in sorted(indexes)]
                for i in deleted_items_u:
                    ind = items_to_delete.index(i)
                    item_user_deletions[ind].append(user_id)
            t = str(len(filtered_list)) + ' ' + ' '.join(filtered_list) + '\n'
            users_item_new.write(t)
        # write the user visits of the deleted articles
        for u, a in zip(item_user_deletions, items_to_delete):
            user_item_test_file.write(str(a))
            user_item_test_file.write(' ')
            user_item_test_file.write(' '.join(str(x) for x in u))
            user_item_test_file.write('\n')
        print "\n" + "number of items deleted to evaluate out matrix prediction : " + str(a) + '\n'


def modify_lda_corpus(lda_corpus_directory, delete_items=1, valid_users=3, test_file=None):
    """
    A function to modify the user item file. This function deletes certain items to detect it using recommendation
    engine.
    :param user_item_directory: the file in which the user item matrix is stored
    :param delete_items: the number of items which has to be deleted
    :param valid_users: the number of items which has to be visited by a valid user
    :return:
    """
    print "Modifying the corpus file for evaluation"
    lda_corpus = io.open(lda_corpus_directory).readlines()
    lda_corpus_new = io.open(lda_corpus_directory, mode='w')
    lda_corpus_test_file = io.open(test_file, mode='w')
    temp = []
    for doc_id, lda_corpus_i in enumerate(lda_corpus):
        lda_corpus_i = lda_corpus_i.split(' ')[:-1]
        indexes = numpy.unique(lda_corpus_i, return_index=True)[1]
        unique_items = [lda_corpus_i[index] for index in sorted(indexes)]
        if len(unique_items) > valid_users:
            items_to_delete = unique_items[-delete_items:]
            for item_to_delete in items_to_delete:
                lda_corpus_i = filter(lambda a: a != item_to_delete, lda_corpus_i)
            temp.append(lda_corpus_i)
            t = str(doc_id) + ' ' + ' '.join(items_to_delete) + '\n'
            lda_corpus_test_file.write(t.decode('unicode-escape'))
        else:
            temp.append(lda_corpus_i)
    for t in temp:
        lda_corpus_new.write(' '.join(t).decode('unicode-escape'))
        lda_corpus_new.write(u'\n')


class GetDataFromMongodb(object):
    """
    A class to extract the data from mongodb.
    """

    def __init__(self, total_number_of_items=5000, total_number_of_users=5000,
                 uri='mongodb://qxuser:userqx16@52.57.103.28/?authSource=users', data_path=None):
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
        self.__user_collection = client.RtlDB.Users
        self.__item_collection = client.RtlDB.Articles
        if self.__no_of_users is None:
            self.__no_of_users = self.__user_collection.find().count()
        if self.__no_of_items is None:
            self.__no_of_items = self.__item_collection.find().count()
        self.__all_users = self.__user_collection.find().sort([['_id', 1]]).limit(self.__no_of_users)
        self.__all_items = self.__item_collection.find().sort([['_id', 1]]).limit(
            self.__no_of_items)  # 1 for ascending, -1 for descending
        self.__data_path = data_path
        self.__user_item = self.__data_path + "user_item.dat"
        self.__user_id = self.__data_path + "user_id.dat"
        self.__item_id = self.__data_path + "item_id.dat"
        self.__item_user = self.__data_path + "item_user.dat"
        self.__corpus = self.__data_path + "corpus.dat"
        self.__vocabulary = self.__data_path + "vocabulary.dat"
        self.__lda_item_id = self.__data_path + "lda_item_id.dat"
        self.__lda_user_id = self.__data_path + "lda_user_id.dat"
        self.__lda_corpus = self.__data_path + "lda_corpus.dat"
        warnings.filterwarnings("ignore")

    def SaveCorpusExtractedFromTfForLDA(self, corpus_location=None):
        """
        A function to save the copus file extracted from the database.

        :param corpus_location: the file location of the corpus, if None it is saved in the data directory
        """
        print "Saving the corpus of the items from the database" + '\n'
        if corpus_location is None:
            corpus_location = self.__corpus
        vocabs = []
        item_doc = io.open(self.__item_id, 'w', encoding='utf-8')
        with io.open(corpus_location, 'w', encoding='utf-8') as file1:
            n = 1
            for datas_all in self.__all_items:
                datas = datas_all.get('tf')
                # check if the tf vector is empty
                if not datas:
                    # print "Empty entry:" + str(datas_all)
                    pass
                else:
                    item_doc.write(datas_all.get('id') + '\n')
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
        with io.open(self.__vocabulary, 'w', encoding='utf-8') as v:
            for vocab in vocabs:
                v.write(vocab + '\n')
        file1.close()
        v.close()
        print "The corpus is saved in the file " + str(self.__corpus) + "\n"

    def SaveCorpusExtractedFromTagsForLDA(self, corpus_location=None):
        """
        A function to save the copus file extracted from the database.

        :param corpus_location: the file location of the corpus, if None it is saved in the data directory
        """
        print "Saving the corpus of the items from the database" + '\n'
        if corpus_location is None:
            corpus_location = self.__corpus
        vocabs = []
        item_doc = io.open(self.__item_id, 'w', encoding='utf-8')
        with io.open(corpus_location, 'w', encoding='utf-8') as file1:
            n = 1
            for datas_all in self.__all_items:
                datas = datas_all.get('tags').split(',')
                # check if the tag is empty
                if len(datas[0]) < 2:
                    pass
                else:
                    item_doc.write(datas_all.get('id') + '\n')
                    # file1.write(str(n).decode("utf-8") + u' en ') # in case of gensim library include this line
                    for data in datas:
                        vocab = data
                        # update vocab file
                        try:
                            i = vocabs.index(data)
                        except:
                            vocabs.append(vocab)
                        # write the data to the corpus file
                        file1.write(vocab + ' ')
                    file1.write(u"\n")
                    n = n + 1
        # write the vocabulary in the vocab file
        with io.open(self.__vocabulary, 'w', encoding='utf-8') as v:
            for vocab in vocabs:
                v.write(vocab + '\n')
        file1.close()
        v.close()
        print "The corpus is saved in the file " + str(corpus_location) + "\n"

    def SaveCorpusExtractedFromUserItemsForLDA(self, filter_users=20, test_recommender=None):
        """
        A function to save the copus file extracted from the user item views database.
        """
        print "Saving the corpus of the user item matrix from the database" + '\n'
        lda_user_id = io.open(self.__lda_user_id, mode='w')
        lda_item_id = io.open(self.__lda_item_id, mode='w')
        lda_corpus = io.open(self.__lda_corpus, mode='w')
        users_db = self.__all_users
        item_ids_unique = []
        for user in users_db:
            user_items = user.get('articles')
            user_id = user.get("id")
            valid_items = []
            for user_item in user_items:
                id = user_item.get('id')
                valid_items.append(id)
            # condition to check the number of valid items, reduces noise
            if len(valid_items) > filter_users:
                lda_user_id.write(user_id)
                # remove the redundant items from the valid items
                indexes = numpy.unique(valid_items, return_index=True)[1]
                unique_valid_items = [valid_items[index] for index in sorted(indexes)]
                # write the item id to the file
                for item in unique_valid_items:
                    if item in item_ids_unique:
                        pass
                    else:
                        item_ids_unique.append(item)
                        t = item + '\n'
                        lda_item_id.write(t.decode('unicode-escape'))
                for item in valid_items:
                    item = str(item) + ' '
                    lda_corpus.write(item.decode('unicode-escape'))
                lda_corpus.write(u'\n')
                lda_user_id.write(u'\n')
        lda_corpus.close()
        lda_user_id.close()
        if test_recommender is True:
            test_file = self.__data_path + "user_item_test_file.dat"
            modify_lda_corpus(lda_corpus_directory=self.__lda_corpus, test_file=test_file, delete_items=1,
                              valid_users=5)

    def SaveDataForCTR(self, user_id_directory=None, user_item_directory=None,
                       item_user_directory=None, item_id_directory=None, test_recommender='in', filter_users=20,
                       valid_unique_articles=10, delete_articles=1):
        """
        A function to save the user_item.dat, item_user.dat, user_id.dat and item_id.dat files to build the
        recommendation engine using collaborative topic regression.

        :param user_id_directory: the file location of the user id information, if None it is saved in the
        data directory
        :param user_item_directory: the file location of the user item information, if None it is saved in the
        data directory
        :param item_user_directory: the file location of the item user information, if None it is saved in the
        data directory
        :param item_id_directory: the file location of the item id information, if None it is saved in the
        data directory
        """
        print "Extracting the user views from the database" + '\n'
        if user_id_directory is None:
            user_id_directory = self.__user_id
        if user_item_directory is None:
            user_item_directory = self.__user_item
        if item_user_directory is None:
            item_user_directory = self.__item_user
        if item_id_directory is None:
            item_id_directory = self.__item_id
        users_db = self.__all_users
        item_ids = io.open(item_id_directory, mode='r').readlines()
        item_ids = map(lambda s: s.strip(), item_ids)
        user_id_file = io.open(user_id_directory, mode='w')
        with io.open(user_item_directory, mode='w') as user_file:
            for user in users_db:
                user_items = user.get('articles')
                user_id = user.get("id")
                valid_items = []
                for user_item in user_items:
                    id = user_item.get('id')
                    if id in item_ids:
                        valid_items.append(id)
                # condition to check the number of valid items, reduces noise
                if len(valid_items) > filter_users:
                    user_id_file.write(user_id)
                    length = str(len(valid_items)).decode('utf-8') + u' '
                    user_file.write(length)
                    for item in valid_items:
                        item = str(item_ids.index(item)).decode('unicode-escape')
                        a_id = item + u' '
                        user_file.write(a_id)
                    user_file.write(u'\n')
                    user_id_file.write(u'\n')
        print "The user item matrix is saved in the file " + str(self.__user_item) + "\n"

        if test_recommender is not None:
            if test_recommender == 'in':
                test_file = self.__data_path + "user_item_test_file.dat"
                copy_file = self.__data_path + "user_item_copy_file.dat"
                modify_user_item(user_item_directory, delete_items=delete_articles, valid_users=valid_unique_articles,
                                 test_file=test_file, user_item_copy_file=copy_file, test_recommender=test_recommender)
            elif test_recommender == 'out':
                test_file = self.__data_path + "item_user_test_file.dat"
                copy_file = self.__data_path + "user_item_copy_file.dat"
                modify_user_item(user_item_directory, delete_items=delete_articles, valid_users=valid_unique_articles,
                                 test_file=test_file, user_item_copy_file=copy_file, test_recommender=test_recommender)

        items_id = io.open(item_id_directory, mode='r').readlines()
        items_id = map(lambda s: s.strip(), items_id)
        temp_items_users = ["", ] * len(items_id)
        users_item = numpy.loadtxt(user_item_directory, dtype=str, delimiter="\n")
        for user_id, user_item in enumerate(users_item):
            user_item = user_item.split(" ")[1:-1]
            for i_user_item in user_item:
                temp_items_users[int(i_user_item)] = str(temp_items_users[int(i_user_item)]) + str(
                    user_id) + " "
        aud = io.open(item_user_directory, mode='w')
        for temp_item_user in temp_items_users:
            aud.write(str(len(temp_item_user.split())).decode('unicode-escape'))
            aud.write(u' ')
            aud.write(temp_item_user.decode('unicode-escape'))
            aud.write(u'\n')
        print "The item user matrix is saved in the file " + str(self.__item_user) + "\n"
