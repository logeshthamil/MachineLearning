import numpy
import pyLDAvis
import lda

matrix_file_location = "/home/lt/Downloads/reducedMatrix_420tags_20users/freqMat.csv"
tag_location = "/home/lt/Downloads/reducedMatrix_420tags_20users/tags.txt"
# user_id_location = "/home/lt/Downloads/tfidf_matrix/userIds.txt"

Print = True

matrix = numpy.loadtxt(matrix_file_location, delimiter=",", dtype=int)
# userId = numpy.loadtxt(user_id_location, delimiter=",", dtype=str)
Tag = numpy.loadtxt(tag_location, delimiter=",", dtype=str)

matrix1 = matrix[:500]
matrix2 = matrix[500:1000]
matrix3 = matrix[:1000]

# userId = userId[:]
Tag = Tag[:]

X = matrix
vocab = Tag

print X.shape

model = lda.LDA(n_topics=10, n_iter=10, alpha=0.1, eta=0.1)
model.fit(X)  # model.fit_transform(X) is also available
if Print is True:
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 11
    for i, topic_dist in enumerate(topic_word):
        topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

topic_term = model.topic_word_  # model.components_ also works
doc_topic = model.doc_topic_
print doc_topic

# doc_lengths = numpy.sum(numpy.transpose(X), axis=0)
# term_frequency = numpy.sum(X, axis=0)
# movies_vis_data = pyLDAvis.prepare(topic_term, doc_topic, doc_lengths, vocab, term_frequency)
# pyLDAvis.save_html(movies_vis_data, "/home/lt/Downloads/lda1.html")
