"""

Code taken from https://github.com/fbickfordsmith/epig/blob/b11124d2dd48381a5756e14d920d401f1fd3120d

"""
import math
import logging

from BASS_utils import logmeanexp

import torch
from torch import Tensor


# taken from ./src/uncertainty/utils.py

def check(
    scores: Tensor,
    min_value: float = 0.0,
    max_value: float = math.inf,
    epsilon: float = 1e-6,
    score_type: str = "",
) -> Tensor:
    """
    Warn if any element of scores is negative, a NaN or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores >= min_value - epsilon) & (scores <= max_value + epsilon)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()

        logging.warning(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")

    return scores


# taken from ./src/uncertainty/bald_probs.py
def entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but more memory-intensive.

    If p(y_i|x) is 0, we make sure p(y_i|x) log p(y_i|x) evaluates to 0, not NaN.

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs: Tensor[float]

    Returns:
        Tensor[float]
    """
    return -torch.sum(torch.xlogy(probs, probs), dim=-1)

def marginal_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim == 3

    probs = torch.mean(probs, dim=1)  # [N, Cl]

    scores = entropy_from_probs(probs)  # [N,]
    scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="ME")  # [N,]

    return scores  # [N,]


# taken from ./src/uncertainty/epig_probs.py

def conditional_epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Estimate the joint predictive distribution.
    probs_pool = probs_pool[:, None, :, :, None]  # [N_p, 1, K, Cl, 1]
    probs_targ = probs_targ[None, :, :, None, :]  # [1, N_t, K, 1, Cl]
    probs_joint = probs_pool * probs_targ  # [N_p, N_t, K, Cl, Cl]
    probs_joint = torch.mean(probs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=2)  # [N_p, 1, Cl, 1]
    probs_targ = torch.mean(probs_targ, dim=2)  # [1, N_t, 1, Cl]

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
    nonzero_joint = probs_joint > 0  # [N_p, N_t, Cl, Cl]
    log_term = torch.clone(probs_joint)  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] = torch.log(probs_joint[nonzero_joint])  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(probs_joint * log_term, dim=(-2, -1))  # [N_p, N_t]

    return scores  # [N_p, N_t]


def epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    _, _, Cl = probs_pool.shape

    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]


def epig_from_probs_using_matmul(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    EPIG(x) = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = H[p(y|x)] + E_{p_*(x_*)}[H[p(y_*|x_*)]] - E_{p_*(x_*)}[H[p(y,y_*|x,x_*)]]

    This uses the fact that I(A;B) = H(A) + H(B) - H(A,B).

    References:
        https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    N_t, K, Cl = probs_targ.shape

    entropy_pool = marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ = marginal_entropy_from_probs(probs_targ)  # [N_t,]

    probs_pool = probs_pool.permute(0, 2, 1)  # [N_p, Cl, K]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_targ = probs_targ.reshape(K, N_t * Cl)  # [K, N_t * Cl]
    probs_joint = probs_pool @ probs_targ / K  # [N_p, Cl, N_t * Cl]

    entropy_joint = -torch.sum(torch.xlogy(probs_joint, probs_joint), dim=(-2, -1)) / N_t  # [N_p,]

    scores = entropy_pool + torch.mean(entropy_targ) - entropy_joint  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]



def epig_from_probs_using_weights(
    probs_pool: Tensor, probs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]
            = ∫ p_*(x_*) EPIG(x|x_*) dx_*
            ~= ∫ p_{pool}(x_*) w(x_*) EPIG(x|x_*) dx_*
            ~= (1 / M) ∑_{i=1}^M w(x_*^i) EPIG(x|x_*^i) where x_*^i in D_{pool}

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]
        weights: Tensor[float], [N_t,]

    Returns:
        Tensor[float], [N_p,]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    _, _, Cl = probs_pool.shape

    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]


# taken from ./src/uncertainty/epig_logprobs.py

def conditional_epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    # Estimate the log of the joint predictive distribution.
    logprobs_pool = logprobs_pool[:, None, :, :, None]  # [N_p, 1, K, Cl, 1]
    logprobs_targ = logprobs_targ[None, :, :, None, :]  # [1, N_t, K, 1, Cl]
    logprobs_joint = logprobs_pool + logprobs_targ  # [N_p, N_t, K, Cl, Cl]
    logprobs_joint = logmeanexp(logprobs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the log of the marginal predictive distributions.
    logprobs_pool = logmeanexp(logprobs_pool, dim=2)  # [N_p, 1, Cl, 1]
    logprobs_targ = logmeanexp(logprobs_targ, dim=2)  # [1, N_t, 1, Cl]

    # Estimate the log of the product of the marginal predictive distributions.
    logprobs_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
    log_term = logprobs_joint - logprobs_joint_indep  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(torch.exp(logprobs_joint) * log_term, dim=(-2, -1))  # [N_p, N_t]

    return scores  # [N_p, N_t]


def epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert logprobs_pool.ndim == logprobs_targ.ndim == 3

    _, _, Cl = logprobs_pool.shape

    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]


def epig_from_logprobs_using_matmul(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    probs_pool = torch.exp(logprobs_pool)  # [N_p, K, Cl]
    probs_targ = torch.exp(logprobs_targ)  # [N_t, K, Cl]

    return epig_from_probs_using_matmul(probs_pool, probs_targ)  # [N_p,]


def epig_from_logprobs_using_weights(
    logprobs_pool: Tensor, logprobs_targ: Tensor, weights: Tensor
) -> Tensor:
    """
    EPIG(x) = I(y;x_*,y_*|x)
            = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = E_{p_*(x_*)}[EPIG(x|x_*)]
            = ∫ p_*(x_*) EPIG(x|x_*) dx_*
            ~= ∫ p_{pool}(x_*) w(x_*) EPIG(x|x_*) dx_*
            ~= (1 / M) ∑_{i=1}^M w(x_*^i) EPIG(x|x_*^i) where x_*^i in D_{pool}

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl], preds on proxy target inputs from the pool
        weights: Tensor[float], [N_t,], weight on each proxy target input

    Returns:
        Tensor[float], [N_p,]
    """
    assert logprobs_pool.ndim == logprobs_targ.ndim == 3

    _, _, Cl = logprobs_pool.shape

    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    scores = weights[None, :] * scores  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]