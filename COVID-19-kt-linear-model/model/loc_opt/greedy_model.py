import numpy as np
from greedy_load_data import DLoader
'''
Greedy model for lockdown policy optimization

Location: (Documented businesses / institutions)
Region: The entire city/state. This model optimizes on a single region at a time.
'''

# Parameters and Inputs

# number of locations
loc_cnt = 0

# Total population of the region to be optimized on
pop = 0

# Number of active cases in the region
I_a = 0

# Number of cumulative cases in the region
I_c = 0

# Betas (probability of infection of a transmission in a location)
# Should be initialized to be a numpy array of shape (loc_cnt,)
betas = np.array([])

# Utility per visitor of each location
# Should be initialized to be a numpy array of shape (loc_cnt,)
util = np.array([])

# Limits(capacity/Maximum number of visitor) of each location
# Should be initialized to be a numpy array of shape (loc_cnt,)
cap = np.array([])

# If this is set to False, a location is either open or closed.
# Otherwise a business can be partially open, meaning that it can still take visitors but have a certain restriction on their number.
enable_partially_open = True


# The threshold of new infections. The new infections resulted from lockdown policy should not exceed this number
infec_max = 100


def prep_data():
    """
    Preprocess data and initialize all the global variables
    :return: None
    """
    loader = DLoader()
    cap = loader.visitor_cnt

    pass




def optimize():
    """
    Optimization Function
    :return: A numpy array of shape (loc_cnt, ) meaning the optimal restriction for each location
    """
    prep_data()

    I_d = 0  # Delta I
    res = np.zeros((loc_cnt,)) # final result
    P = (I_a/pop) * (1 - 1/pop)
    pr = P*betas

    stop = False


    # While there's room for more relaxation
    while np.sum((cap-res) > 0) and I_d < infec_max and not stop:
        stop = True
        n_mat = np.repeat(res[None], loc_cnt, axis=0) + np.diag(np.ones(loc_cnt,))
        n_mat = n_mat * np.clip(n_mat - 1, 0, None)
        I_change = util / ((n_mat @ pr.T) - I_d).T        # Utility change / Infection Changes
        order = np.argsort(I_change)
        for i in order:
            if res[i] < cap[i] and I_d + I_change < infec_max:
                res[i] = res[i] + 1
                I_d += I_change[i]
                stop = False
                break
    return res


if __name__ == '__main__':
    prep_data()
    ans = optimize()
    # print(ans)