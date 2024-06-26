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

def epsilon_ma_vec(qs0, alpha, sigma):
    qs = np.array(qs0)
    data_ind = alpha / (sigma ** 2)
    mu2 = np.multiply(sigma, np.sqrt(np.log(np.divide(1, qs))))
    mu1 = np.add(mu2, 1)
    e1 = np.divide(mu1, (sigma ** 2))
    e2 = np.divide(mu2, (sigma ** 2))
    Ahat = np.pow(np.multiply(qs, np.exp(e2)), np.divide(np.subtract(mu2, 1), mu2))
    A = np.divide(np.subtract(1, qs), np.subtract(1, Ahat))
    B = np.exp(e1) / np.pow(qs, np.divide(1, np.subtract(mu1, 1)))
    data_dep = np.multiply(1 / (alpha - 1),
                           np.log(np.add(
                               np.multiply(np.subtract(1, qs), np.pow(A, (alpha - 1))),
                               np.multiply(qs, np.pow(B, (alpha - 1)))))
                           )
    tot = 0
    # tot = []
    for it in range(len(qs)):  # check conditions
        if 0 < qs[it] < 1 < mu2[it] and qs[it] <= math.exp((mu2[it] - 1) * e2[it]) / (
                mu1[it] * mu2[it] / ((mu1[it] - 1) * mu2[it] - 1)) and mu1[it] >= alpha:
            tot += min(float(data_dep[it]), data_ind)
            # tot.append(min(float(data_dep[it]), data_ind))
        else:
            tot += data_ind
        # tot.append(data_ind)
    return tot

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
    tot = 0
    for k in range(2,alpha + 1):
        comb = scipy.special.comb(alpha,k)
        tot += comb * ((1-p)**(alpha-k))*(p**k)*math.exp((k-1)*k/(sigma1**2))
    logarand = ((1-p)**(alpha-1))*(1+(alpha-1)*p)+tot
    eprime = 1/(alpha-1) * math.log(logarand)
    rdp_epsilon = K * eprime + epsilon_ma(qs, alpha, sigma2)
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