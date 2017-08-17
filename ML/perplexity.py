import numpy
import scipy.special
import os
from sklearn.preprocessing import normalize

try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp
import io

chunk_location = "./temp.txt"


def dirichlet_expectation(alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.
    """
    if (len(alpha.shape) == 1):
        result = scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha))
    else:
        result = scipy.special.psi(alpha) - scipy.special.psi(numpy.sum(alpha, 1))[:, numpy.newaxis]
    return result.astype(alpha.dtype)  # keep the same precision as input


def bound(corpus, _lambda=None, _gamma=None, subsample_ratio=1.0, alpha=None, topics=None, beta=None):
    """
    Estimate the variational bound of documents from `corpus`:
    E_q[log p(corpus)] - E_q[log q(corpus)]
    `gamma` are the variational parameters on topic weights for each `corpus`
    document (=2d matrix=what comes out of `inference()`).
    If not supplied, will be inferred from the model.
    """
    alpha = [alpha, ] * topics
    eta = [[beta], ] * topics
    score = 0.0
    gamma = _gamma
    Elogbeta = dirichlet_expectation(_lambda)
    chunksize = len(corpus)
    for d, doc in enumerate(corpus):  # stream the input doc-by-doc, in case it's too large to fit in RAM
        # if d % chunksize == 0:
        #     print "bound: at document #" +str(d)
        gammad = gamma[d]
        Elogthetad = dirichlet_expectation(gammad)
        # E[log p(doc | theta, beta)]
        score += numpy.sum(cnt * logsumexp(Elogthetad + Elogbeta[:, id]) for id, cnt in doc)
        # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
        score += numpy.sum((alpha - gammad) * Elogthetad)
        score += numpy.sum(scipy.special.gammaln(gammad) - scipy.special.gammaln(alpha))
        score += scipy.special.gammaln(numpy.sum(alpha)) - scipy.special.gammaln(numpy.sum(gammad))
    # compensate likelihood for when `corpus` above is only a sample of the whole corpus
    score *= subsample_ratio
    # E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
    score += numpy.sum((eta - _lambda) * Elogbeta)
    score += numpy.sum(scipy.special.gammaln(_lambda) - scipy.special.gammaln(eta))
    if numpy.ndim(eta) == 0:
        sum_eta = eta * len(corpus)
    else:
        sum_eta = numpy.sum(eta, 1)
    score += numpy.sum(scipy.special.gammaln(sum_eta) - scipy.special.gammaln(numpy.sum(_lambda, 1)))
    return score


def log_perplexity(chunk, alpha=None, beta=None, _lambda=None, _gamma=None):
    """
    Calculate and return per-word likelihood bound, using the `chunk` of
    documents as evaluation corpus. Also output the calculated statistics. incl.
    perplexity=2^(-bound), to log at INFO level.
    """
    total_docs = len(chunk)
    corpus_words = sum(cnt for document in chunk for _, cnt in document)
    subsample_ratio = 1.0 * total_docs / len(chunk)
    b = bound(chunk, subsample_ratio=subsample_ratio, alpha=alpha, beta=beta, _lambda=_lambda, _gamma=_gamma,
              topics=_lambda.shape[0])
    p = (subsample_ratio * corpus_words)
    perwordbound = b / p
    return perwordbound


def corpus_to_chunk(corpus, vocabulary):
    """
    Transform the corpus into chunk data.
    :param corpus: the corpus location
    :param vocabulary: the vocabulary file location
    :param chunk: the chunk data location
    """
    chunk_file = io.open(chunk_location, mode='w', encoding='utf-8')
    corpus = io.open(corpus, encoding='utf-8', mode='r').readlines()
    vocab = io.open(vocabulary, encoding='utf-8', mode='r').readlines()
    vocab = [s[:-1] for s in vocab]
    for c_line in corpus:
        chunk = {}
        c_line_ind = c_line.split(' ')
        for c_line_word_ind in c_line_ind:
            if c_line_word_ind in vocab:
                if c_line_word_ind in chunk.values():
                    pass
                else:
                    count = c_line_ind.count(c_line_word_ind)
                    index = vocab.index(c_line_word_ind) + 1
                    temp = {index: count}
                    chunk.update(temp)
        for p in chunk.items():
            chunk_file.write(u"%r,%r " % p)
        chunk_file.write(u"\n")


def chunk_to_desired():
    """
    Transform the chunk data into the desired format for perplexity calculation.
    :param chunk_location: the chunk location
    :return: the data in desired format
    """
    temp_corpus = []
    corpus_chunk = numpy.loadtxt(chunk_location, delimiter="\n", dtype=str)
    for i_chunk in corpus_chunk:
        i_chunk = i_chunk[:-1].split(" ")
        temp_temp_corpus = []
        for i_elem in i_chunk:
            temp_temp_corpus.append(map(int, i_elem.split(',')))
        temp_corpus.append(temp_temp_corpus)
    os.remove(chunk_location)
    return temp_corpus


def calculate_perplexity(corpus=None, vocab=None, _lambda=None, _gamma=None, _alpha=None, _beta=None):
    corpus_to_chunk(corpus=corpus, vocabulary=vocab)
    chunk = chunk_to_desired()
    _lambda = numpy.loadtxt(_lambda, dtype=float, delimiter=" ")
    _gamma = numpy.loadtxt(_gamma, dtype=float, delimiter=" ")
    _lambda[_lambda == 0] = 1 ** -15
    _gamma[_gamma == 0] = 1 ** -15
    _gamma = normalize(_gamma, norm='l1', axis=1)
    _lambda = normalize(_lambda, norm='l1', axis=0)
    perwb = log_perplexity(chunk=chunk, alpha=_alpha, beta=_beta, _lambda=_lambda, _gamma=_gamma)
    print "per word bound: " + str(perwb)
    print "perplexity: " + str(2 ** (-perwb))
