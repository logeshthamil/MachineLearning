import os, time, subprocess


class CollaborativeTopicRegression(object):
    """
    A class to generate the make file of Collaborative Topic Regression (c++ implementation) and provide the inputs to
    it and get the output
    """

    def __init__(self, a=1, b=0.1, lambda_u=0.01, lambda_v=100, number_of_iterations=100, number_of_factors=500,
                 data_directory=None):
        """
        :param a: the confidence value of found ratings
        :param b: the confidence value of not found ratings
        :param lambda_u: the regularization parameter lambda u
        :param lambda_v: the regularization parameter lambda v
        :param number_of_iterations: the total number of iterations
        :param number_of_factors: the number of factors
        :param data_directory: the output directory
        """
        script_path = os.path.abspath(__file__)
        data_directory = data_directory
        ctr_directory = os.path.join(os.path.split(script_path)[0][:-12], "source_code/ctr/")
        self.__a = a
        self.__b = b
        self.__lambda_u = lambda_u
        self.__lambda_v = lambda_v
        self.__number_of_iterations = number_of_iterations
        self.__number_of_factors = number_of_factors
        self.__op_directory = data_directory + "ctr_outputs/"
        self.__user_item = data_directory + "user_item.dat"
        self.__item_user = data_directory + "item_user.dat"
        self.__mult = data_directory + "mult.dat"
        self.__gamma = data_directory + "gamma.dat"
        self.__lambda = data_directory + "lambda.dat"
        self.__ctr_location = ctr_directory
        self.__ldagamma = data_directory + "lda_gamma.dat"
        self.__ldalambda = data_directory + "lda_lambda.dat"
        self._compile_ctr(ctr_source_location=self.__ctr_location)

    def _compile_ctr(self, ctr_source_location=None):
        """
        Compile the source of the collaborative topic regression which is written in c++.
        :param ctr_source_location: the location of the source code of ctr
        """
        if ctr_source_location is not None:
            self.__ctr_location = ctr_source_location
        print "compiling the source code of ctr"
        os.chdir(self.__ctr_location)
        os.system("make")
        print

    def GetOutput(self):
        """
        Save the output of the collaborative topic regression in the output location.
        """
        os.chdir(self.__ctr_location)
        print "Applying the ctr technique on the output of the LDA" + "\n"
        cmd = ['./ctr', '--directory', self.__op_directory, '--user', self.__user_item, '--item', self.__item_user,
               '--mult', self.__mult, '--theta_init', self.__gamma, '--beta_init', self.__lambda, '--a', str(self.__a),
               '--b', str(self.__b), '--lambda_u', str(self.__lambda_u), '--lambda_v', str(self.__lambda_v),
               '--max_iter', str(self.__number_of_iterations), '--num_factors', str(self.__number_of_factors),
               '--save_lag', '100']
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for line in p.stdout:
            pass
        p.wait()
        print("Collaborative filtering: time taken for computation: %0.3f seconds." % (time.time() - t0))

    def GetOutput_withoutusingLDA(self):
        """
        Save the output of the collaborative topic regression in the output location.
        """
        os.chdir(self.__ctr_location)
        print "Applying the ctr technique on the user item matrix" + "\n"
        cmd = ['./ctr', '--directory', self.__op_directory[:-1], '--user', self.__user_item, '--item', self.__item_user,
               '--a', str(self.__a), '--b', str(self.__b), '--lambda_u', str(self.__lambda_u), '--lambda_v',
               str(self.__lambda_v),
               '--max_iter', str(self.__number_of_iterations), '--num_factors', str(self.__number_of_factors),
               '--save_lag', '100']
        t0 = time.time()
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        for line in p.stdout:
            pass
        p.wait()
        print("Collaborative filtering: time taken for computation: %0.3f seconds." % (time.time() - t0))
