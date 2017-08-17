import io, numpy


def corpus_to_frequencymatrix(corpus, vocabulary):
    """
    Convert the corpus into the frequency matrix.

    :param corpus: the corpus file location
    :param vocabulary: the vocabulary file location
    :return: the frequency matrix
    """
    vocabulary = [line.rstrip('\n') for line in io.open(vocabulary, encoding='utf8')]
    corpus = io.open(corpus, encoding='utf8').readlines()
    freqmat = numpy.zeros((len(corpus), len(vocabulary)))
    for line_number, corpus_line in enumerate(corpus):
        corpus_line_words = corpus_line.split(" ")[:-1]
        for corpus_word in corpus_line_words:
            try:
                p = vocabulary.index(corpus_word)
                freqmat[line_number, p] = freqmat[line_number, p] + 1
            except:
                pass
    return freqmat
