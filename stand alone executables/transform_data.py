import numpy
from sklearn.preprocessing import normalize


def output_to_desired(doc_topic_location=None, topic_word_location=None, total_topic_location=None):
    """
    Transform the output of lda which is unnormalized into the desired format with normalized values.

    :param doc_topic_location: output location where doc topic is saved
    :param topic_word_location: output location where topic word is saved
    :param total_topic_location: output location where the total count of documents in topics is saved
    :return: lambda and gamma
    """
    doc_topic = numpy.loadtxt(doc_topic_location, delimiter="\n", dtype=str)
    topic_word = numpy.loadtxt(topic_word_location, delimiter="\n", dtype=str)
    total_topic = numpy.loadtxt(total_topic_location, delimiter=" ", dtype=str)[1:]
    no_of_topics = len(total_topic)
    no_of_docs = len(doc_topic)
    no_of_words = len(topic_word)
    doc_topic_numpy = numpy.zeros((no_of_docs, no_of_topics))
    topic_word_numpy = numpy.zeros((no_of_topics, no_of_words))
    for doc_number, i_chunk in enumerate(doc_topic):
        i_chunk = i_chunk.split(" ")[2:]
        for i_i_chunk in i_chunk:
            topic, weight = i_i_chunk.split(":")
            doc_topic_numpy[doc_number, topic] = int(weight)
    for word_number, i_word in enumerate(topic_word):
        i_word = i_word.split(" ")[1:]
        for i_i_word in i_word:
            topic, weight = i_i_word.split(":")
            topic_word_numpy[topic, word_number] = int(weight)

    # normalize
    doc_topic_numpy_norm = normalize(doc_topic_numpy, norm='l1', axis=1)
    topic_word_numpy_norm = normalize(topic_word_numpy, norm='l1', axis=0)

    # dont normalize
    # doc_topic_numpy_norm = doc_topic_numpy
    # topic_word_numpy_norm = topic_word_numpy

    # replace zero value with minimum value
    # doc_topic_numpy_norm[doc_topic_numpy_norm == 0] = 1 ** -15
    # topic_word_numpy_norm[topic_word_numpy_norm == 0] = 1 ** -15

    return doc_topic_numpy_norm, topic_word_numpy_norm


def write_numpynd_to_file(numpy_array=None, file_location=None):
    """
    Write the numpy array to a text file.

    :param numpy_array: the input numpy array
    :param file_location: the file location
    :return: None
    """
    numpy.savetxt(fname=file_location, X=numpy_array)


def main():
    """
    example:    python transform_data.py --dt "/home/lt/PycharmProjects/Recommendation/data/doc_topic.0"
    --tw "/home/lt/PycharmProjects/Recommendation/data/server_0_table_0.model"
    --dto "./temp_gamma.txt"
    --two "./temp_lambda.txt"
    --tt "/home/lt/PycharmProjects/Recommendation/data/server_0_table_1.model"
    """
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("--dt", dest="doc_topic", type="str", help="get document topic distribution unnormalized")
    parser.add_option("--tw", dest="topic_word", type="str", help="get topic word distribution unnormalized")
    parser.add_option("--tt", dest="total_topic", type="str", help="get total topic counts")
    parser.add_option("--dto", dest="doc_topic_out", type="str",
                      help="get location for normalized document topic distribution")
    parser.add_option("--two", dest="topic_word_out", type="str",
                      help="get location for normalized topic word distribution")
    (options, args) = parser.parse_args()
    print options.doc_topic
    print options.topic_word
    print options.doc_topic_out
    print options.topic_word_out
    _gamma, _lambda = output_to_desired(doc_topic_location=options.doc_topic,
                                        topic_word_location=options.topic_word,
                                        total_topic_location=options.total_topic)
    write_numpynd_to_file(numpy_array=_gamma, file_location=options.doc_topic_out)
    write_numpynd_to_file(numpy_array=_lambda, file_location=options.topic_word_out)


if __name__ == "__main__":
    main()
