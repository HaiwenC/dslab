from model.linear_model.multi_si_linear import *
import numpy as np


data_4 = np.array([[1, 2, 3, 4, 5, 6], [3, 5, 6, 43, 87, 223], [4, 5, 6, 7, 8, 9]])
popu = np.array([1555, 2333, 4444])
pf = np.array([[23, 23, 154], [43, 432, 45], [22, 11, 667]])
k = 2
jp = 1
alpha = 0.1
T_tr = 4
horizon = 10


def main():
    arr = np.array([[1,2,3],[4,5,6],[7,8,9]])
    arr2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    tt = np.arange(10,0,-1)
    tt = np.power(3,tt)
    tt = np.array([tt.T,]*10)

def test_pre_data():
    # test passed


    x,y = data_prep(data_4, pf, popu,k,jp)
    # print(x)
    # print(y)
    return x,y

def test_ind_beta():
    x, y = data_prep(data_4, pf, popu, k, jp)
    betas = ind_beta(x,y,alpha, k, T_tr, popu,jp)
    return betas

def test_simulate_pred(betas):
    infections = simulate_pred(data_4,pf,betas,popu,k,horizon, jp)
    print("res")
    print(infections)


if __name__ == '__main__':
    p = 1
    for i in range(23):
        p = p * ((365.0 - i) / 365)
    print(1.0 - p)