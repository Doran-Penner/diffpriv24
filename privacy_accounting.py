import math
import numpy as np
import scipy

def epsilon_ma(qs, alpha, sigma):
    """
    This is a function to calculate the data-dependent or data-independent renyi epsilon
    in a data-dependent way. Refer to Tory's algorithm for this please.
    :param qs: list of q values for each uniquely answered query. Don't ask me
               what a q value is.
    :param alpha: integer representing the order of the renyi-divergence
    :param sigma: float representing the standard deviation for the noise used in GNMax
    :returns: float representing tot, a sum of the total privacy budget spent.
    """
    data_ind = alpha / (sigma ** 2)
    tot = 0
    # tot = []
    for q in qs:
        if not (0 < q < 1):
            tot += data_ind
            continue
        mu2 = sigma * math.sqrt(math.log(1 / q))
        mu1 = mu2 + 1
        e1 = mu1 / (sigma ** 2)
        e2 = mu2 / (sigma ** 2)
        Ahat = math.pow(q * math.exp(e2), (mu2 - 1) / mu2)
        A = (1 - q) / (1 - Ahat)
        B = math.exp(e1) / math.pow(q, 1 / (mu1 - 1))
        data_dep = 1 / (alpha - 1) * math.log((1 - q) * (A ** (alpha - 1)) + q * (B ** (alpha - 1)))
        # check conditions
        if 0 < q < 1 < mu2 and q <= math.exp((mu2 - 1) * e2) / (mu1 * mu2 / ((mu1 - 1) * mu2 - 1)) and mu1 >= alpha:
            tot += min(data_dep, data_ind)
            # tot.append(min(data_dep, data_ind))
        else:
            tot += data_ind
            # tot.append(data_ind)
    return tot

def single_epsilon_ma(q, alpha, sigma):
    """
    Function used to calculate the ma epsilon for renyi differential privacy for a single query
    :param q: float representing the q value for a single query
    :param alpha: numeric representing the alpha value for the order of the renyi differential
                  privacy
    :param sigma: float representing the scale value for the normal distribution used when
                  reporting the results of a novel query
    :returns: float representing the additional ma epsilon cost incurred by a specific query
    """
    data_ind = alpha / (sigma ** 2)
    if not 0 < q < 1:
        return data_ind
    mu2 = sigma * math.sqrt(math.log(1 / q))
    mu1 = mu2 + 1
    e1 = mu1 / (sigma ** 2)
    e2 = mu2 / (sigma ** 2)
    Ahat = math.pow(q * math.exp(e2), (mu2 - 1) / mu2)
    A = (1 - q) / (1 - Ahat)
    B = math.exp(e1) / math.pow(q, 1 / (mu1 - 1))
    data_dep = 1 / (alpha - 1) * math.log((1 - q) * (A ** (alpha - 1)) + q * (B ** (alpha - 1)))
    # check conditions
    if 0 < q < 1 < mu2 and q <= math.exp((mu2 - 1) * e2) / (mu1 * mu2 / ((mu1 - 1) * mu2 - 1)) and mu1 >= alpha:
        return min(data_dep, data_ind)
    else:
        return data_ind
 
def epsilon_ma_vec(qs, alpha, sigma):
    """
    Function that (i think) does the same as epsilon_ma but in a vectorized way, granting
    it additional speed. TODO : get response from Doran about what this function is

    :param qs: list of q values for each uniquely answered query. Don't ask me
               what a q value is.
    :param alpha: integer representing the order of the renyi-divergence
    :param sigma: float representing the standard deviation for the noise used in GNMax
    :returns: float representing tot, a sum of the total privacy budget spent.
    """
    qs = np.asarray(qs)
    # we need to use data-indep for q outside of (0,1), but just ignoring q values
    # outside of that will cause a ZeroDivision error or some other problem
    # thus we modify those q values and save which ones were fine to begin with
    safe_qs = (0 < qs) & (qs < 1)  # this'll go into the final "indep vs dep" check
    qs[np.logical_not(safe_qs)] = 0.5  # setto dummy value which won't cause errors

    data_ind = alpha / (sigma ** 2)
    mu2 = sigma * np.sqrt(np.log(1 / qs))
    mu1 = mu2 + 1
    e1 = mu1 / (sigma ** 2)
    e2 = mu2 / (sigma ** 2)
    Ahat = np.pow(qs * np.exp(e2), (mu2 - 1) / mu2)
    A = (1 - qs) / (1 - Ahat)
    B = np.exp(e1) / np.pow(qs, (1 / (mu1 - 1)))
    data_dep = (
        1
        / (alpha - 1)
        * np.log((1 - qs) * (A ** (alpha - 1)) + qs * (B ** (alpha - 1)))
    )

    best = np.minimum(data_dep, data_ind)
    
    can_use_data_dep = (
        (safe_qs)
        & (1 < mu2)
        & (qs <= (np.exp((mu2 - 1) * e2) / (mu1 * mu2 / ((mu1 - 1) * mu2 - 1))))
        & (mu1 >= alpha)
    )
    # np.where(conds, xs, ys) iterates through all arrays and returns array of
    # x in xs when cond in conds is true and y in ys otherwise
    tot = np.where(can_use_data_dep, best, data_ind)
    return np.sum(tot)

def renyi_to_ed(epsilon, delta, alpha):
    """
    Function to convert from renyi-differential privacy to epsilon delta-differential privacy, given some delta value.
    :param epsilon: float representing the renyi epsilon value
    :param delta: float representing what delta you want in the conversion (the lower the delta, the higher the epsilon)
    :param alpha: float representing the renyi alpha value
    :returns: float representing epsilon for epsilon-delta differential privacy
    """
    A = max((alpha-1)*epsilon - math.log(delta * alpha/((1-1/alpha)**(alpha-1))),0)
    B = math.log((math.exp((alpha-1)*epsilon)-1)/(alpha*delta)+1)
    return 1/(alpha - 1) * min(A,B)

def epsilon_prime(alpha, p, sigma1):
    """
    Function used to calculate the epsilon prime value used in calculating
    renyi differential privacy of RepeatGNMax

    :param alpha: numeric representing the order of the order of the renyi
                  differential privacy
    :param p: float used to parametrize the poisson subsampling in Repeat-
              GNMax
    :param sigma1: float used to parametrize the noise added when comparing
                   voting records in RepeatGNMax
    :returns: float representing the epsilon prime value for calculating
              renyi differential privacy
    """
    tot = 0
    for k in range(2,alpha + 1):
        comb = scipy.special.comb(alpha,k)
        tot += comb * ((1-p)**(alpha-k))*(p**k)*math.exp((k-1)*k/(sigma1**2))
    logarand = ((1-p)**(alpha-1))*(1+(alpha-1)*p)+tot
    eprime = 1/(alpha-1) * math.log(logarand)
    return eprime

def repeat_epsilon(qs, K, alpha, sigma1, sigma2, p, delta):
    """
    Function to calculate the epsilon for RepeatGNMax, given some delta value.
    :param qs: list of q values for each uniquely answered query. Don't ask me
               what a q value is.
    :param K: integer representing the total number of queries answered
    :param alpha: int representing the order of the renyi divergence
    :param sigma1: float representing the noise used when comparing the current
                   voting record to the older voting records
    :param sigma2: float representing the amount of noise used when releasing the
                   results of queries that have new results
    :param p: float representing the p value for poisson subsampling when subsampling
              teachers to compare to previous queries
    :param delta: float representing the desired delta for epsilon delta differential
                  privacy
    :returns: float representing the epsilon for the epsilon-delta differential privacy
              of the mechanism
    """
    eprime = epsilon_prime(alpha, p)
    eps = epsilon_ma_vec(qs, alpha, sigma2)
    print("first term:", K*eprime)
    print("second term:",eps)
    rdp_epsilon = K * eprime + eps
    return renyi_to_ed(rdp_epsilon, delta, alpha)

def gnmax_epsilon(qs, alpha, sigma, delta):
    """
    Function to calculate the epsilon for GNMax, given some delta value.
    :param qs: list of q values for each uniquely answered query
    :param alpha: int representing the order of the renyi divergence
    :param sigma: float representing the standard deviation of the gaussian noise used
    :param delta: float representingthe delta corresponding to the desired epsilon value
    :returns: float representing the calculated epsilon value
    """
    return renyi_to_ed(epsilon_ma(qs,alpha,sigma),delta,alpha)
