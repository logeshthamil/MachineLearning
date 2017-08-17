import source_code, multiprocessing


# TODO: use multithreading to execute all the functions simultaneously
# TODO: remove the hard coded part from all the functions and place it outside
# TODO: PLSA recommender - implement and evaluate

def get_recommendation_only_LDA():
    # input parameters or user preference
    total_number_of_items = 15000
    total_number_of_users = 2000000
    number_of_topics = 40
    iterations_lda = 1000
    alpha = 0.1
    beta = 0.01
    test_recommender = 'in'
    valid_users = 20

    # data directory
    data_directory = "/home/lt/PycharmProjects/Recommendation/Recommendation_Engine_using_CTR/data/data2/"

    # connect mongodb database with GetData block and save the data for LDA and CTR operations
    get_data = source_code.GetDataFromMongodb(total_number_of_items=total_number_of_items,
                                              total_number_of_users=total_number_of_users, data_path=data_directory)
    get_data.SaveCorpusExtractedFromUserItemsForLDA(filter_users=valid_users, test_recommender=test_recommender)

    # Light lda usage
    lda = source_code.LightLDA_for_recommendation(number_of_topics=number_of_topics, iterations=iterations_lda,
                                                  alpha=alpha, beta=beta, data_path=data_directory)
    lda.GetOutput()

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender, only_lda=True)


def fullcycle_lightlda_tf_in_matrix():
    # input parameters or user preference
    test_recommender = 'in'
    delete_articles = 1

    # data directory
    data_directory = "/home/lt/quanox/QX_Recommendations/recommendation-python/Recommendation_Engine_using_CTR/data/data4/"

    # connect mongodb database with GetData block and save the data for LDA and CTR operations
    # get_data = source_code.GetDataFromMongodb(total_number_of_items=total_number_of_items,
    #                                           total_number_of_users=total_number_of_users,
    #                                           data_path=data_directory)
    # # get_data.SaveCorpusExtractedFromTagsForLDA()
    # get_data.SaveCorpusExtractedFromTfForLDA()
    # get_data.SaveDataForCTR(test_recommender=test_recommender, filter_users=valid_users,
    #                         valid_unique_articles=valid_unique_articles, delete_articles=delete_articles)

    # Light lda usage
    lda = source_code.LightLDA(number_of_topics=number_of_topics, iterations=iterations_lda, alpha=alpha, beta=beta,
                               data_path=data_directory)
    lda.GetOutput()

    # compute CTR on the saved data and get output
    ctr = source_code.CollaborativeTopicRegression(a=a, b=b, lambda_u=lambda_u, lambda_v=lambda_v,
                                                   number_of_iterations=iterations_ctr, number_of_factors=no_of_factors,
                                                   data_directory=data_directory)
    ctr.GetOutput()

    print "Getting recommendation using LDA" + '\n'

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)

    print "Getting recommendation without using LDA" + '\n'
    ctr.GetOutput_withoutusingLDA()

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)


def fullcycle_lightlda_tags_in_matrix():
    # input parameters or user preference
    test_recommender = 'in'
    delete_articles = 1

    # data directory
    data_directory = "/home/lt/quanox/QX_Recommendations/recommendation-python/Recommendation_Engine_using_CTR/data/data3/"

    # connect mongodb database with GetData block and save the data for LDA and CTR operations
    # get_data = source_code.GetDataFromMongodb(total_number_of_items=total_number_of_items,
    #                                           total_number_of_users=total_number_of_users,
    #                                           data_path=data_directory)
    # get_data.SaveCorpusExtractedFromTagsForLDA()
    # # get_data.SaveCorpusExtractedFromTfForLDA()
    # get_data.SaveDataForCTR(test_recommender=test_recommender, filter_users=valid_users,
    #                         valid_unique_articles=valid_unique_articles, delete_articles=delete_articles)

    # Light lda usage
    lda = source_code.LightLDA(number_of_topics=number_of_topics, iterations=iterations_lda, alpha=alpha, beta=beta,
                               data_path=data_directory)
    lda.GetOutput()

    # compute CTR on the saved data and get output
    ctr = source_code.CollaborativeTopicRegression(a=a, b=b, lambda_u=lambda_u, lambda_v=lambda_v,
                                                   number_of_iterations=iterations_ctr, number_of_factors=no_of_factors,
                                                   data_directory=data_directory)
    ctr.GetOutput()

    print "Getting recommendation using LDA" + '\n'

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)

    print "Getting recommendation without using LDA" + '\n'
    ctr.GetOutput_withoutusingLDA()

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)


def fullcycle_lightlda_tags_out_matrix():
    # input parameters or user preference
    test_recommender = 'out'  # 'in' or 'out'
    delete_articles = 100

    # data directory
    data_directory = "/tmp/qualitytest/ctrold/"

    # connect mongodb database with GetData block and save the data for LDA and CTR operations
    get_data = source_code.GetDataFromMongodb(total_number_of_items=total_number_of_items,
                                              total_number_of_users=total_number_of_users,
                                              data_path=data_directory)
    get_data.SaveCorpusExtractedFromTagsForLDA()
    # get_data.SaveCorpusExtractedFromTfForLDA()
    get_data.SaveDataForCTR(test_recommender=test_recommender, filter_users=valid_users,
                            valid_unique_articles=valid_unique_articles, delete_articles=delete_articles)

    # Light lda usage
    lda = source_code.LightLDA(number_of_topics=number_of_topics, iterations=iterations_lda, alpha=alpha, beta=beta,
                               data_path=data_directory)
    lda.GetOutput()

    # compute CTR on the saved data and get output
    ctr = source_code.CollaborativeTopicRegression(a=a, b=b, lambda_u=lambda_u, lambda_v=lambda_v,
                                                   number_of_iterations=iterations_ctr, number_of_factors=no_of_factors,
                                                   data_directory=data_directory)
    ctr.GetOutput()

    print "Getting recommendation using LDA" + '\n'

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)

    print "Getting recommendation without using LDA" + '\n'
    ctr.GetOutput_withoutusingLDA()

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)


def fullcycle_lightlda_tf_out_matrix():
    # input parameters or user preference
    test_recommender = 'out'  # 'in' or 'out'
    delete_articles = 100

    # data directory
    data_directory = "/home/lt/quanox/QX_Recommendations/recommendation-python/Recommendation_Engine_using_CTR/data/data2/"
    #
    # # connect mongodb database with GetData block and save the data for LDA and CTR operations
    # get_data = source_code.GetDataFromMongodb(total_number_of_items=total_number_of_items,
    #                                           total_number_of_users=total_number_of_users,
    #                                           data_path=data_directory)
    # # get_data.SaveCorpusExtractedFromTagsForLDA()
    # get_data.SaveCorpusExtractedFromTfForLDA()
    # get_data.SaveDataForCTR(test_recommender=test_recommender, filter_users=valid_users,
    #                         valid_unique_articles=valid_unique_articles, delete_articles=delete_articles)

    # Light lda usage
    lda = source_code.LightLDA(number_of_topics=number_of_topics, iterations=iterations_lda, alpha=alpha, beta=beta,
                               data_path=data_directory)
    lda.GetOutput()

    # compute CTR on the saved data and get output
    ctr = source_code.CollaborativeTopicRegression(a=a, b=b, lambda_u=lambda_u, lambda_v=lambda_v,
                                                   number_of_iterations=iterations_ctr, number_of_factors=no_of_factors,
                                                   data_directory=data_directory)
    ctr.GetOutput()

    print "Getting recommendation using LDA" + '\n'

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)

    print "Getting recommendation without using LDA" + '\n'
    ctr.GetOutput_withoutusingLDA()

    # get recommendation from the output of the CTR
    recommendation = source_code.GetRecommendation(data_path=data_directory)
    recommendation.SaveRecommendation(test_recommender=test_recommender)


# input parameters or user preference
total_number_of_items = 10000
total_number_of_users = 100000
number_of_topics = 25
iterations_lda = 1000
alpha = 0.1
beta = 0.01
a = 1
b = 0.1
lambda_u = 0.01
lambda_v = 100
iterations_ctr = 10
no_of_factors = 10
valid_users = 20
valid_unique_articles = 5
#
# p1 = multiprocessing.Process(target=fullcycle_lightlda_tags_out_matrix)
# p2 = multiprocessing.Process(target=fullcycle_lightlda_tf_out_matrix)
# p3 = multiprocessing.Process(target=fullcycle_lightlda_tags_in_matrix)
# p4 = multiprocessing.Process(target=fullcycle_lightlda_tf_in_matrix)
#
# jobs = [p1, p2, p3, p4]
#
# p1.start()
# p2.start()
# p3.start()
# p4.start()

fullcycle_lightlda_tags_out_matrix()
# fullcycle_lightlda_tf_out_matrix()
# fullcycle_lightlda_tags_in_matrix()
# fullcycle_lightlda_tf_in_matrix()
# get_recommendation_only_LDA()
