import numpy as np


def prediction(evidence_data_add, prior, start_day, end_day):
    # you need to implement this method.

    x_prob_rain = []
    # x_prob_sunny[i] = 1 - x_prob_rain[i]

    f = open(evidence_data_add, "r")
    lines = f.readlines()
    for line in lines:
        take = 'take umbrella' in line
        rain_prob = np.array([[0.7, 0.3], [0.3, 0.7]])
        if take:
            u_r_p = np.array([[0.9, 0.0], [0.0, 0.2]])
            be = np.matmul(prior, rain_prob)
            temp = np.matmul(be, u_r_p)
            temp_sum = 1/(temp[0] + temp[1])
            temp[0] = temp[0] * temp_sum
            temp[1] = temp[1] * temp_sum
            prior = [temp[0], temp[1]]
        else:
            u_s_p = np.array([[0.1, 0.0], [0.0, 0.8]])
            be = np.matmul(prior, rain_prob)
            temp = np.matmul(be, u_s_p)
            temp_sum = 1 / (temp[0] + temp[1])
            temp[0] = temp[0] * temp_sum
            temp[1] = temp[1] * temp_sum
            prior = [temp[0], temp[1]]
        x_prob_rain.append(prior[0])
    #print(x_prob_rain[99])

    for i in range(start_day, end_day+1):
        rain_last = x_prob_rain[i-2]
        sun_last = 1 - x_prob_rain[i-2]
        r_after = 0.7*rain_last + 0.3*sun_last
        prior_new = [r_after, 1-r_after]
        x_prob_rain.append(prior_new[0])

    return x_prob_rain


# following lines are main function:
evidence_data_add = "data//assign2_umbrella.txt"
start_day = 101
end_day = 150
# the prior distribution on the initial state, P(X0). 50% rainy, and 50% sunny on day 0.
prior = [0.5, 0.5]

if __name__=="__main__":
    x_prob_rain=prediction(evidence_data_add, prior, start_day, end_day)
    for i in range(start_day, end_day+1):
        # print("Day " + str(i) + ": rain " + str(x_prob_rain[i-start_day]) + ", sunny " + str(1 - x_prob_rain[i-start_day]))
        print("Day " + str(i) + ": rain " + str(x_prob_rain[i-1]) + ", sunny " + str(1 - x_prob_rain[i-1]))