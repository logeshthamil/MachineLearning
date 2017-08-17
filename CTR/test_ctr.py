import numpy
import ML

u_file = "/home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data/final-U.dat"
v_file = "/home/lt/PycharmProjects/Recommendation/data/ctr_blei/rtl_data/final-V.dat"
recommendation = ML.ctr.GetRecommendation(u_file=u_file, v_file=v_file)
print recommendation.GetRecommendation(user_id=2350)
print recommendation.GetRecommendation(user_id=2352)
print recommendation.GetRecommendation(user_id=2353)
