import numpy


def compute_covariance_matrix(array1=None, array2=None):
    """
    Function to compute the covariance matrix.

    :param array1: the first array
    :param array2: the second array
    :return: the covariance matrix
    """
    x = array1
    if array2 is None:
        y = array1
    else:
        y = array2
    z = numpy.vstack((x, y))
    c = numpy.cov(z.T)
    return c


def principal_components(matrix, percentage):
    """
    Function to reduce the dimensionality of a matrix using principal component analysis. Only top n singular values are
    used for matrix synthesis. The n singular values are selected based on the percentage of energy which reside in it.

    :param matrix: the input data matrix
    :param percentage: the percentage of energy (used to reduce the noise)
    :return: the dimensionality reduced matrix
    """
    u, s, v = numpy.linalg.svd(matrix)
    sum_singular_values = numpy.sum(s)
    temp = 0
    index = len(s)
    for i, s_value in enumerate(s):
        temp = temp + s_value
        if numpy.sum(temp) >= sum_singular_values * percentage:
            index = i
            break
    reduced_s = s[:index]
    diag_s = numpy.zeros((len(reduced_s), len(reduced_s)))
    numpy.fill_diagonal(diag_s, reduced_s)
    reduced_u = u[:, range(len(reduced_s))]
    reduced_matrix_output = numpy.dot(reduced_u, diag_s)
    return reduced_matrix_output


def matrixtocoordinates(matrix_file_location):
    """
    Find the coordinates to cluster from the matrix file.

    :param matrix_file_location: the matrix file location
    :return: the array of coordinates
    """
    matrix = numpy.loadtxt(matrix_file_location, delimiter=",", dtype=int)
    matrix = matrix[:6590]
    shape = matrix.shape
    coordinates = []
    for x in range(shape[0]):
        for y in range(shape[1]):
            if matrix[x][y] != 0:
                to_append = [[x, y], ] * matrix[x][y]
                coordinates.extend(to_append)
    return numpy.asarray(coordinates[:])
