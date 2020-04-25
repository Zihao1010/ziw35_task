import numpy as np
import pandas as pd # I want to read the txt file as dataframe

def get_p_b_cd():
    # you need to implement this method.
    p_b_cd = np.zeros((3, 3, 2), dtype=np.float)
    for c in range(1, 4):
        for d in range(1, 3):
            for b in range(1, 4):
                sum_total = sum((file["c"] == c) & (file["d"] == d))
                sum_b = sum((file["b"] == b) & (file["c"] == c) & (file["d"] == d))
                p_b_cd[b-1, c-1, d-1] = sum_b / sum_total
    return p_b_cd


def get_p_a_be():
    # you need to implement this method.
    p_a_be = np.zeros((2, 3, 2), dtype=np.float)
    for b in range(1, p_a_be.shape[1]+1):
        for e in range(1, p_a_be.shape[2]+1):
            for a in range(1, p_a_be.shape[0] + 1):
                sum_total = sum((file["b"] == b) & (file["e"] == e))
                sum_a = sum((file["a"] == a) & (file["b"] == b) & (file["e"] == e))
                p_a_be[a - 1, b - 1, e - 1] = sum_a / sum_total
    return p_a_be


# following lines are main function:
data_add = "data//assign2_BNdata.txt"

file = pd.read_csv(data_add, sep='\s+')

if __name__ == '__main__':


    # probability distribution of b.
    p_b_cd = get_p_b_cd()
    for c in range(3):
        for d in range(2):
            for b in range(3):
                print("P(b=" + str(b + 1) + "|c=" + str(c + 1) + ",d=" + str(d + 1) + ")=" + str(p_b_cd[b][c][d]))

    # probability distribution of a.
    p_a_be = get_p_a_be()
    for b in range(3):
        for e in range(2):
            for a in range(2):
                print("P(a=" + str(a + 1) + "|b=" + str(b  + 1) + ",e=" + str(e + 1) + ")=" + str(p_a_be[a][b][e]))
