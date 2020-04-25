import numpy as np


def filtering(evidence_data_add, prior, total_day):
    # you need to implement this method.
    x_prob_rain = []

    # x_prob_sunny[i] = 1 - x_prob_rain[i]

    f = open(evidence_data_add, "r")
    lines = f.readlines()
    if total_day <= 100:# avoid the total_day number is larger than 100
        for line in lines[0:total_day]:
            take = 'take umbrella' in line
            rain_prob = np.array([[0.7, 0.3], [0.3, 0.7]])
            if take:
                u_r_p = np.array([[0.9, 0.0], [0.0, 0.2]])
                be = np.matmul(prior, rain_prob)
                temp = np.matmul(be, u_r_p)
                temp_sum = 1/(temp[0] + temp[1])# get the value of alpha.
                temp[0] = temp[0] * temp_sum
                temp[1] = temp[1] * temp_sum
                prior = [temp[0], temp[1]]#change the value of prior every time
            else:
                u_s_p = np.array([[0.1, 0.0], [0.0, 0.8]])
                be = np.matmul(prior, rain_prob)
                temp = np.matmul(be, u_s_p)
                temp_sum = 1 / (temp[0] + temp[1])
                # print(temp_sum)
                temp[0] = temp[0] * temp_sum
                temp[1] = temp[1] * temp_sum
                prior = [temp[0], temp[1]]
            x_prob_rain.append(prior[0])#put the value of probility of rain in the list
            # print(x_prob_rain)
    return x_prob_rain


# following lines are main function:
if __name__=="__main__":
    evidence_data_add = "data//assign2_umbrella.txt"
    total_day = 100
# the prior distribution on the initial state, P(X0). 50% rainy, and 50% sunny on day 0.
    prior = [0.5, 0.5]

    x_prob_rain=filtering(evidence_data_add, prior, total_day)
    for i in range(100):
        print("Day " + str(i+1) + ": rain " + str(x_prob_rain[i]) + ", sunny " + str(1 - x_prob_rain[i]))