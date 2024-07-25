import math
import numpy as np
import scipy


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
    # NOTE if q = 1/e^(1/sigma^2) then we get a "divide by zero WARNING" but it still runs so it's okay?
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
    Function that does the same as epsilon_ma but in a vectorized way, granting
    it additional speed.
    :param qs: list of q values for each uniquely answered query. Don't ask me
               what a q value is.
    :param alpha: integer representing the order of the renyi-divergence
    :param sigma: float representing the standard deviation for the noise used in GNMax
    
    returns: float representing tot, a sum of the total privacy budget spent.
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

def epsilon_prime_swing(alpha, p, sigma1):
    """
    Incorrect function used to calculate the epsilon prime for the swing voter
    metric. Carter says that it's wrong. I don't doubt him.

    :param alpha: numeric representing the order of the order of the renyi
                  differential privacy
    :param p: float used to parametrize the poisson subsampling in Repeat-
              GNMax
    :param sigma1: float used to parametrize the noise added when comparing
                   voting records in RepeatGNMax
    
    :returns: float representing the epsilon prime value for calculating
              renyi differential privacy
    """
    # hacky: be careful of this if we change any functions
    return epsilon_prime(alpha, p, sigma1 * math.sqrt(2))

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
    return renyi_to_ed(epsilon_ma_vec(qs,alpha,sigma),delta,alpha)

def heterogeneous_strong_composition(epsilons, deltas, delta_prime):
    """
    A function to calculate the strong composition of multiple differentially private
    queries based on the composition theorems for heterogeneous mechanisms in
    http://proceedings.mlr.press/v37/kairouz15.pdf
    
    :param epsilons: numpy array of the epsilon values
    :param deltas: numpy array of the corresponding delta values
    :param delta_prime: float for the calculation of the final delta value

    :returns: tuple containing new_epsilon, new_delta
    """

    # first we can calculate the new delta value

    # this probably isn't optimized but it will get the job done for now
    # do 1-delta for each delta
    for i in range(len(deltas)):
        deltas[i] = 1 - deltas[i]
    
    # 1 - (1-delta_prime)product(1-delta_i)
    new_delta = 1 - ((1 - delta_prime) * np.prod(deltas))


    # now for the new value of epsilon, we have to take the minimum of three possible
    # values, which we will compute in the same order as they come up in the paper

    eps_val_1 = np.sum(epsilons)


    # noew we are gonna calculate the additive term out front (to be used in the 
    # second and third possible epsilon values)
    # also, we will calculate the sum of squares for the epsilon since that get used
    eps_prefix = 0
    eps_squared = 0
    for epsilon in epsilons:
        eps_prefix += (math.exp(epsilon) - 1) * epsilon / (math.exp(epsilon) + 1)
        eps_squared += epsilon ** 2

    eps_val_2 = eps_prefix + math.sqrt(2* eps_squared * math.log(math.e + math.sqrt(eps_squared)/ delta_prime))
    
    eps_val_3 = eps_prefix + math.sqrt(2*eps_squared * math.log(1/delta_prime))

    new_epsilon = min(eps_val_1,eps_val_2,eps_val_3)

    return new_epsilon,new_delta
  
def homogeneous_strong_composition(epsilon,delta,delta_prime, k):
    """
    A function to calculate the strong composition of multiple differentially private
    queries based on the composition theorems for homogeneous mechanisms in
    http://proceedings.mlr.press/v37/kairouz15.pdf
    
    :param epsilon: float to represent the epsilon value
    :param delta: float to represent the delta value
    :param delta_prime: float for the calculation of the final delta value
    :param k: number of mechanisms to compose
    
    :returns: tuple containing new_epsilon, new_delta
    """

    new_delta = 1 - ((1 - delta_prime) * ((1 - delta) ** k))

    eps_prefix = epsilon * k * (math.exp(epsilon) - 1) / (math.exp(epsilon) + 1)

    new_epsilon = min(
            k * epsilon,
            eps_prefix + epsilon * math.sqrt(2 * k * math.log(math.e + epsilon * math.sqrt(k) / delta_prime)),
            eps_prefix + epsilon * math.sqrt(2 * k * math.log(1 / delta_prime))
        )

    return new_epsilon, new_delta

def laplacian_eps_prime(p, epsilon, delta = 0):
    """
    a function to calculate the epsilon prime of the laplacian repeat mechanism
    This is based on the amplification by subsampling https://arxiv.org/pdf/2210.00597

    :param p: a float that comes from the poisson parameter of the aggregator
    :param epsilon: the epsilon cost of the mechanism without subsampling
    :param delta: the delta cost of the mechanism without subsampling

    :returns: tuple containing epsilon_prime, delta_prime
    """

    # does this need to be its own function? no, but consistency I guess
    
    eps_prime = math.log(1 + p * (math.exp(epsilon) - 1))
    delta_prime = delta * p

    return eps_prime, delta_prime
