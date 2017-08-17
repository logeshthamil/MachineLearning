import numpy
import hdplda2

modified_matrix_file = "/home/lt/quanox/data/freqMat.lda.vw"
tags = "/home/lt/quanox/data/tags.txt"
alpha = 0.817
gamma = 0.122
beta = 0.500
iteration = 100

modified_matrix = numpy.loadtxt(modified_matrix_file, dtype=dict, delimiter=" ")[:500]
tags = numpy.loadtxt(tags, dtype=str, delimiter="\n")

modified_matrix = modified_matrix[:, 1:-1]

hdplda_matrix = []

for entry_row in modified_matrix:
    temp = numpy.zeros(0)
    for entry_column in entry_row:
        entry_column = entry_column.split(':')
        tag_number = int(entry_column[0])
        tag_count = int(entry_column[1])
        entry = [tag_number, ] * tag_count
        temp = numpy.append(temp, entry)
    hdplda_matrix.append(temp.astype(int).tolist())
hdplda_matrix = numpy.asarray(hdplda_matrix)
print hdplda_matrix

# f = open('lda_input.csv', 'ab')
# for i in hdplda_matrix:
#     numpy.savetxt(f, i, fmt='%i')

# algorithm = hdplda2
# hdplda = algorithm.HDPLDA(alpha, amma, beta, hdplda_matrix, len(tags))
# # hdplda.dump()
#
# import cProfile
#
# # cProfile.runctx('algorithm.hdplda_learning(hdplda, iteration)', globals(), locals(), 'algorithm.hdplda.profile')
# algorithm.hdplda_learning(hdplda, iteration)
# # output_summary(hdplda, voca)
#
# word_dist = []
# for d in hdplda.worddist():
#     word_dist.append(d.values())
#
# word_dist = numpy.asarray(word_dist)
# doc_dist = numpy.asarray(hdplda.docdist())
#
# # numpy.savetxt("/home/lt/quanox/data/word_dist", word_dist)
# numpy.savetxt("/home/lt/quanox/data/doc_dist", doc_dist)
