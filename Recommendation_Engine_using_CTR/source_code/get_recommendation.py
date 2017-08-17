import numpy, os, io, random


class GetRecommendation(object):
    """
    Get the Recommendation from the output collaborative topic regression by reforming the rating matrix from the
    u and v matrix files.
    """

    def __init__(self, u_file=None, v_file=None, no_of_recommendations=10, output_file=None, data_path=None):
        """
        :param u_file: the u matrix file location
        :param v_file: the v matrix file location
        :param no_of_recommendations: the number of recommendations to be generated
        :param output_file: the output file in which the recommendations should be saved
        :param data_path: the path in which the data can be found
        """
        if u_file is None:
            u_file = os.path.join(data_path, "ctr_outputs/final-U.dat")
            self.__u = numpy.loadtxt(u_file, delimiter=" ", dtype=str)
        else:
            self.__u = numpy.loadtxt(u_file, delimiter=" ", dtype=str)
        if v_file is None:
            v_file = os.path.join(data_path, "ctr_outputs/final-V.dat")
            self.__v = numpy.loadtxt(v_file, delimiter=" ", dtype=str)
        else:
            self.__v = numpy.loadtxt(v_file, delimiter=" ", dtype=str)
        if output_file is None:
            self.__recommendation_file = os.path.join(data_path, "ctr_outputs/Recommendation.dat")
        else:
            self.__recommendation_file = output_file
        self.__data_path = data_path
        self._r = self.__GetR()
        self.__no_of_rec = no_of_recommendations

    def __GetR(self):
        """
        Form the ratings matrix by multiplying the u and v matrix.
        :return: the ratings matrix
        """
        u = self.__u
        v = self.__v
        u = u[0:-1, 0:-1]
        v = v[0:-1, 0:-1]
        u = u.astype(numpy.float)
        v = v.astype(numpy.float)
        v = numpy.transpose(v)
        r = numpy.dot(u, v)
        return r

    def GetRecommendation_for_user(self, no_of_recommendations=20, user_id=None):
        """
        Get the recommendation by finding the top rated items from the ratings matrix.
        :param no_of_recommendations: the total number of recommendation to be generated
        :param user_id: the user id for which the recommendation should be generated
        :return: the array with the index of the recommended items
        """
        if user_id is None:
            print "Please give the user id to get recommendation"
        else:
            if no_of_recommendations is not None:
                self.__no_of_rec = no_of_recommendations
            numpy.set_printoptions(threshold=numpy.inf)
            recommendations = self._r[user_id].argsort()[-self.__no_of_rec:][::-1]
            return recommendations

    def GetRandomRecommendation_for_user(self, no_of_recommendations=20, user_id=None):
        """
        Get the recommendation by finding the top rated items from the ratings matrix.
        :param no_of_recommendations: the total number of recommendation to be generated
        :param user_id: the user id for which the recommendation should be generated
        :return: the array with the index of the recommended items
        """
        if user_id is None:
            print "Please give the user id to get recommendation"
        else:
            if no_of_recommendations is not None:
                self.__no_of_rec = no_of_recommendations
            shape = self._r.shape[1]
            total_users = range(shape)
            random_recommendation = random.sample(total_users, self.__no_of_rec)
            return random_recommendation

    def GetRecommendation_for_article(self, no_of_recommendations=20, article_id=None):
        """
        Get the recommendation by finding the top rated items from the ratings matrix.
        :param no_of_recommendations: the total number of recommendation to be generated
        :param user_id: the user id for which the recommendation should be generated
        :return: the array with the index of the recommended items
        """
        if article_id is None:
            print "Please give the article id to get recommendation"
        else:
            if no_of_recommendations is not None:
                self.__no_of_rec = no_of_recommendations
            numpy.set_printoptions(threshold=numpy.inf)
            t_r = numpy.transpose(self._r)
            recommendations = t_r[article_id].argsort()[-self.__no_of_rec:][::-1]
            return recommendations, t_r.shape[1]

    def GetRandomRecommendation_for_article(self, no_of_recommendations=20, article_id=None):
        """
        A method to get the random recommendation for the articles.
        :param no_of_recommendations: the total number of recommendation to be generated
        :param user_id: the user id for which the recommendation should be generated
        :return: the array with the index of the recommended items
        """
        if article_id is None:
            print "Please give the article id to get recommendation"
        else:
            if no_of_recommendations is not None:
                self.__no_of_rec = no_of_recommendations
            numpy.set_printoptions(threshold=numpy.inf)
            t_r = numpy.transpose(self._r)
            shape = t_r.shape[0]
            total_articles = range(shape)
            random_recommendation = random.sample(total_articles, self.__no_of_rec)
            return random_recommendation, t_r.shape[1]

    def SaveRecommendation(self, user_id_file=None, item_id_file=None, test_recommender=True, only_lda=False):
        """
        Save the recommendations to a file.
        :param user_id_file: the user id file
        :param item_id_file: the item id file
        """
        print "\n" + "saving recommendation"
        if only_lda is True:
            user_id_file = os.path.join(self.__data_path, "lda_user_id.dat")
            item_id_file = os.path.join(self.__data_path, "lda_item_id.dat")
        else:
            if user_id_file is None:
                user_id_file = os.path.join(self.__data_path, "user_id.dat")
            if item_id_file is None:
                item_id_file = os.path.join(self.__data_path, "item_id.dat")
        r = self.__GetR()
        item_ids = io.open(item_id_file, mode='r').readlines()
        item_ids = map(lambda s: s.strip(), item_ids)
        user_ids = io.open(user_id_file, mode='r').readlines()
        user_ids = map(lambda s: s.strip(), user_ids)
        no_of_users = r.shape[0] + 1
        no_of_items = r.shape[1] + 1
        assert no_of_users == len(user_ids), no_of_items == len(item_ids)
        recommendation_file = io.open(self.__recommendation_file, mode='w')
        for i in range(no_of_users - 1):
            user_id = str(user_ids[i]) + ' '
            recommendation_file.write(user_id.decode('unicode-escape'))
            recommended_items = self.GetRecommendation_for_user(user_id=i)
            # print i, recommended_items
            for r in recommended_items:
                item_id = str(item_ids[r]) + ' '
                recommendation_file.write(item_id.decode('unicode-escape'))
            recommendation_file.write(u'\n')
        if test_recommender == 'in':
            print "\n" + "Evaluating recommendation engine - Inmatrix prediction" + '\n'
            test_data_file = os.path.join(self.__data_path, "user_item_test_file.dat")
            test_data = io.open(test_data_file, mode='r').readlines()
            a = []
            ran_a = []
            items_length_test = 0
            exception_data = []
            for d in test_data:
                d = d.split(' ')
                id = d[0]
                articles = d[1:]
                articles[-1] = articles[-1].strip()
                items_length_test = items_length_test + len(articles)
                try:
                    rec = self.GetRecommendation_for_user(user_id=int(id), no_of_recommendations=50)
                    ran_rec = self.GetRandomRecommendation_for_user(no_of_recommendations=50, user_id=int(id))
                    rec = numpy.asarray(rec)
                    ran_rec = numpy.asarray(ran_rec)
                    articles = numpy.asarray(articles)
                    for i in rec:
                        if i == int(articles[0]):
                            a.append(id)
                    for i in ran_rec:
                        if i == int(articles[0]):
                            ran_a.append(id)
                except:
                    exception_data.append(len(test_data))
            print "Number of exception datas: " + str(len(exception_data)) + '\n'
            print "The total hits from the recommendation: " + str(len(a)) + '\n'
            print "Accuracy in predicting users, recommendation engine: " + str((float(len(a)) / items_length_test)
                                                                                * 100) + '\n'
            print "Accuracy in predicting users, random recommender: " + str(
                (float(len(ran_a)) / items_length_test) * 100)
        elif test_recommender == 'out':
            print "\n" + "Evaluating recommendation engine - Outmatrix evaluation" + '\n'
            test_data_file = os.path.join(self.__data_path, "item_user_test_file.dat")
            test_data = io.open(test_data_file, mode='r').readlines()
            a = []
            a_ran = []
            users_length_test = 0
            recommendations_num = 0
            users_no = 0
            for d in test_data:
                d = d.split(' ')
                a_id = d[0]
                users = d[1:]
                users[-1] = users[-1].strip()
                users_length_test = users_length_test + len(users)
                recommendations_no = len(users) * 1
                try:
                    rec, no_users = self.GetRecommendation_for_article(article_id=int(a_id),
                                                                       no_of_recommendations=recommendations_no)
                    random_rec, no_users = self.GetRandomRecommendation_for_article(article_id=int(a_id),
                                                                                    no_of_recommendations=recommendations_no)
                    rec = numpy.asarray(rec)
                    random_rec = numpy.asarray(random_rec)
                    for i in rec:
                        if str(i).decode('unicode-escape') in users:
                            a.append(str(i))
                    for i in random_rec:
                        if str(i).decode('unicode-escape') in users:
                            a_ran.append(str(i))
                    recommendations_num = recommendations_num + recommendations_no
                    users_no = users_no + no_users
                except:
                    pass
            print "The total hits from the recommendation to articles: " + str(len(a)) + '\n'
            print "The total hits by a random recommender: " + str(len(a_ran)) + '\n'
            print "Accuracy in predicting users, recommendation engine: " + str(
                (float(len(a)) / users_length_test) * 100) + '\n'
            print "Accuracy in predicting users, random recommender: " + str(
                (float(len(a_ran)) / users_length_test) * 100) + '\n'
