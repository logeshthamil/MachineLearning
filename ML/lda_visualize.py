import pyLDAvis
import io, numpy
from sklearn.preprocessing import normalize
import ML.corpus_to_freqMat


def visualize_lda(corpus=None, doc_topic_mat=None, topic_word_mat=None, vocabulary_file=None,
                  output_file="/home/lt/Downloads/lda.html"):
    """
    Visualize the output of lda using clusters in a webpage

    :param corpus: the entire corpus file
    :param doc_topic_mat: the doc topic probablity matrix file (op of lda)
    :param topic_word_mat: the topic word matrix file (op of lda)
    :param vocabulary: the vocabulary file
    :return: save the output of lda as clusters in the output file
    """
    freqmat = ML.corpus_to_freqMat.corpus_to_frequencymatrix(corpus=corpus, vocabulary=vocabulary_file)
    vocabulary = [line.rstrip('\n') for line in io.open(vocabulary_file, encoding='utf8')]
    doc_topic_mat = numpy.loadtxt(doc_topic_mat).astype(float)
    topic_word_mat = numpy.loadtxt(topic_word_mat)
    doc_topic_mat[doc_topic_mat == 0] = 1 ** -15
    topic_word_mat[topic_word_mat == 0] = 1 ** -15
    doc_topic_mat = normalize(doc_topic_mat, norm='l1', axis=1)
    topic_word_mat = normalize(topic_word_mat, norm='l1', axis=0)
    movies_vis_data = pyLDAvis.prepare(topic_term_dists=topic_word_mat,
                                       doc_topic_dists=doc_topic_mat,
                                       doc_lengths=numpy.sum(freqmat, axis=1),
                                       vocab=vocabulary,
                                       term_frequency=numpy.sum(freqmat, axis=0),
                                       mds='tsne')
    pyLDAvis.save_html(movies_vis_data, output_file)
    print "Please find the output html file in the location : " + output_file
