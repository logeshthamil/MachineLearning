import os

ctr_directory = "/home/lt/PycharmProjects/Recommendation/CTR/github_lda/ctr"
input_directory = "/home/lt/PycharmProjects/Recommendation/data/ctr_blei/inputs/"
output_directory = "/home/lt/PycharmProjects/Recommendation/data/ctr_blei/inputs/temp"
_lambda_filename = "final.beta"
_gamma_filename = "final.gamma"
user_data_filename = "users.dat"
item_data_filename = "items.dat"
article_term_filename = "mult.dat"
alpha = 0.01
beta = 0.1
lambda_u = 0.01
lambda_v = 100
num_of_factors = 25

user_directory = input_directory + user_data_filename
item_directory = input_directory + item_data_filename
lambda_directory = input_directory + _lambda_filename
gamma_directory = input_directory + _gamma_filename
article_term_directory = input_directory + article_term_filename

print ctr_directory
os.chdir(ctr_directory)
ctr_command1 = "./ctr " + "--directory " + output_directory + " --user " + user_directory + " --item " + item_directory + \
               " --mult " + article_term_directory + " --theta_init " + gamma_directory + " --beta_init " + lambda_directory \
               + " --a " + str(alpha) + " --b " + str(beta) + " --lambda_u " + str(lambda_u) + " --lambda_v " + str(
    lambda_v) \
               + " --num_factors " + str(num_of_factors)
ctr_command2 = "./ctr --directory /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp --user " \
               "/home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/users.dat " \
               "--item /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/items.dat " \
               "--mult /home/lt/ClionProjects/github_lda/ctr/data/temp_data/mult.dat " \
               "--theta_init /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/final.gamma " \
               "--beta_init /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/final.beta " \
               "--a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 --num_factors 25"
print ctr_command1
print ctr_command2
os.system(ctr_command2)

# ./ctr --directory /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp --user /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/users.dat --item /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/items.dat --mult /home/lt/ClionProjects/github_lda/ctr/data/temp_data/mult.dat --theta_init /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/final.gamma --beta_init /home/lt/ClionProjects/github_lda/ctr/data/temp_data/temp/final.beta --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 --num_factors 25
# ./ctr --directory /home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data --user /home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data/users.dat --item /home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data/items.dat --mult /home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data/corpus.dat --theta_init /home/lt/PycharmProjects/Recommendation/data/lightlda/gamma.dat --beta_init /home/lt/PycharmProjects/Recommendation/data/lightlda/lambda.dat --a 1 --b 0.01 --lambda_u 0.01 --lambda_v 100 --num_factors 25
