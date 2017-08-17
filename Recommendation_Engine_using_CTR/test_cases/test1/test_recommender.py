import source_code


def fullcycle_lightlda_tags():
    """
    Compare the two recommendation engines (LDA + CF, CF) using the in matrix prediction evaluation. In this evaluation
    the tags of the articles are used to construct the corpus.
    """
    # input parameters or user preference
    total_number_of_items = 10000
    total_number_of_users = 1900000
    number_of_topics = 20
    iterations_lda = 1000
    alpha = 0.1
    beta = 0.01
    a = 1
    b = 0.1
    lambda_u = 0.01
    lambda_v = 100
    iterations_ctr = 1
    no_of_factors = 100
    test_recommender = True
    valid_users = 20
    valid_unique_articles = 5
    delete_articles = 1

    # data directory
    data_directory = "/home/lt/quanox/QX_Recommendations/recommendation-python/Recommendation_Engine_using_CTR/data/data2/"

    # connect mongodb database with GetData block and save the data for LDA and CTR operations
    get_data = source_code.GetDataFromMongodb(total_number_of_items=total_number_of_items,
                                              total_number_of_users=total_number_of_users,
                                              data_path=data_directory)
    get_data.SaveCorpusExtractedFromTagsForLDA()
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


def fullcycle_lightlda_tf():
    """
    Compare the two recommendation engines (LDA + CF, CF) using the in matrix prediction evaluation. In this evaluation
    the term frequency of the articles are used to construct the corpus.
    """
    # input parameters or user preference
    total_number_of_items = 10000
    total_number_of_users = 1900000
    number_of_topics = 20
    iterations_lda = 1000
    alpha = 0.1
    beta = 0.01
    a = 1
    b = 0.1
    lambda_u = 0.01
    lambda_v = 100
    iterations_ctr = 1
    no_of_factors = 100
    test_recommender = True
    valid_users = 20
    valid_unique_articles = 5
    delete_articles = 1

    # data directory
    data_directory = "/home/lt/quanox/QX_Recommendations/recommendation-python/Recommendation_Engine_using_CTR/data/data2/"

    # connect mongodb database with GetData block and save the data for LDA and CTR operations
    get_data = source_code.GetDataFromMongodb(total_number_of_items=total_number_of_items,
                                              total_number_of_users=total_number_of_users,
                                              data_path=data_directory)
    get_data.SaveCorpusExtractedFromTfForLDA()
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


fullcycle_lightlda_tags()
fullcycle_lightlda_tf()
