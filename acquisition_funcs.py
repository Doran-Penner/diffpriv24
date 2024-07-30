"""
Acquisition functions to be used in Active Learning!

For now this will only have BALD and BatchBALD

code taken (with minimal change) from:
https://github.com/IzzyBrand/PyTorch_BatchBALD/blob/master

batchBALD based on the paper:
https://arxiv.org/pdf/1906.08158

with BALD originally from:
https://arxiv.org/pdf/1112.5745
"""
import torch
import math
import globals
import numpy as np
from itertools import combinations_with_replacement


def H(x, eps=1e-6):
    """ Compute the element-wise entropy of x

    Arguments:
        x {torch.Tensor} -- array of probabilities in (0,1)

    Keyword Arguments:
        eps {float} -- prevent failure on x == 0

    Returns:
        torch.Tensor -- H(x)
    """
    return -(x+eps)*torch.log(x+eps)

def class_combinations(c, n, m=np.inf):
    """ Generates an array of n-element combinations where each element is one of
    the c classes (an integer). If m is provided and m < n^c, then instead of all
    n^c combinations, m combinations are randomly sampled.

    Arguments:
        c {int} -- the number of classes
        n {int} -- the number of elements in each combination

    Keyword Arguments:
        m {int} -- the number of desired combinations (default: {np.inf})

    Returns:
        np.ndarry -- An [m x n] or [n^c x n] array of integers in [0, c)
    """

    if m < c**n:
        # randomly sample combinations
        return np.random.randint(c, size=(int(m), n))
    else:
        p_c = combinations_with_replacement(np.arange(c), n)
        return np.array(list(iter(p_c)), dtype=int)


class Acquirer:
    """
    Base class for other acquisition functions
    """

    def __init__(self,batch_size,dat_obj,device=globals.device):
        # dat_obj should be the pool we are choosing from (or subset thereof)
        self.batch_size = batch_size
        self.dat_obj = dat_obj
        self.device = device
        self.processing_batch_size = 64 # changed for consistency?

    def score(self,model,x):
        return torch.zeros(len(x))

    def select_batch(self,model,pool_data):

        pool_loader = torch.utils.data.DataLoader(pool_data,
            batch_size=self.processing_batch_size, pin_memory=True, shuffle=False)
        scores = torch.zeros(len(pool_data)).to(self.device)
        for batch_idx, (data, _) in enumerate(pool_loader):
            end_idx = batch_idx + data.shape[0]
            scores[batch_idx:end_idx] = self.score(model, data.to(self.device))

        best_local_indices = torch.argsort(scores)[-self.batch_size:]
        best_global_indices = np.array(pool_data.indices)[best_local_indices.cpu().numpy()]
        return best_global_indices
    

class BALD(Acquirer):
    """
    non-batch version of BALD as a class, a mixture of both linked repos
    """

    def __init__(self,batch_size,dat_obj,device=globals.device):
        super().__init__(batch_size,dat_obj,device)
    

    def score(self,model,x,mc_samps=100):
        model.eval()
        # make sure we are testing and not learning!
        with torch.no_grad():
            # get a bunch of probability vectors for x
            Y = torch.stack([model(x) for i in range(mc_samps)], dim=1)
            # compute the entropy stuff
            H1 = H(Y.mean(axis=1)).sum(axis=1)
            H2 = H(Y).sum(axis=(1,2))/mc_samps

            return H1 - H2


class RandAcquire(Acquirer):

    def __init__(self, batch_size,dat_obj, device=globals.device):
        super().__init__(batch_size,dat_obj, device)


    def score(self, model, _):
        return np.random.rand()


class BatchBALD(Acquirer):
    def __init__(self, batch_size,dat_obj, device=globals.device,subset_size=500):
        super(BatchBALD, self).__init__(batch_size, dat_obj, device)
        self.m = 1e4  # number of MC samples for label combinations
        self.num_sub_pool = subset_size  # number of datapoints in the subpool from which we acquire

    def select_batch(self, model, pool_data, k=100):
        # I(y;W | x) = H1 - H2 = H(y|x) - E_w[H(y|x,W)]

        c = 10 # number of classes

        # performing BatchBALD on the whole pool is very expensive, so we take
        # a random subset of the pool.
        num_extra = len(pool_data) - self.num_sub_pool
        if num_extra > 0:
            sub_pool_data, _ = torch.utils.data.random_split(pool_data, [self.num_sub_pool, num_extra])
        else:
            # even if we don't have enough data left to split, we still need to
            # call random_splot to avoid messing up the indexing later on
            sub_pool_data, _ = torch.utils.data.random_split(pool_data, [len(pool_data), 0])

         # forward pass on the pool once to get class probabilities for each x
        with torch.no_grad():
            pool_loader = torch.utils.data.DataLoader(sub_pool_data,
                batch_size=self.processing_batch_size, pin_memory=True, shuffle=False)
            pool_p_y = torch.zeros(len(sub_pool_data), c, k)
            for batch_idx, (data, _) in enumerate(pool_loader):
                end_idx = batch_idx + data.shape[0]
                pool_p_y[batch_idx:end_idx] = torch.stack([model(data.to(self.device)) for i in range(k)], dim=1).permute(0,2,1)

        # this only need to be calculated once so we pull it out of the loop
        H2 = (H(pool_p_y).sum(axis=(1,2))/k).to(self.device)

        # get all class combinations
        c_1_to_n = class_combinations(c, self.batch_size, self.m)

        # tensor of size [m x k]
        p_y_1_to_n_minus_1 = None

        # store the indices of the chosen datapoints in the subpool
        best_sub_local_indices = []
        # create a mask to keep track of which indices we've chosen
        remaining_indices = torch.ones(len(sub_pool_data), dtype=bool).to(self.device)
        for n in range(self.batch_size):
            # tensor of size [N x m x l]
            p_y_n = pool_p_y[:, c_1_to_n[:, n], :].to(self.device)
            # tensor of size [N x m x k]
            p_y_1_to_n = torch.einsum('mk,pmk->pmk', p_y_1_to_n_minus_1, p_y_n)\
                if p_y_1_to_n_minus_1 is not None else p_y_n

            # and compute the left entropy term
            H1 = H(p_y_1_to_n.mean(axis=2)).sum(axis=1)
            # scores is a vector of scores for each element in the pool.
            # mask by the remaining indices and find the highest scoring element
            scores = H1 - H2
            # print(scores)
            best_local_index = torch.argmax(scores - np.inf*(~remaining_indices)).item()
            # print(f'Best idx {best_local_index}')
            best_sub_local_indices.append(best_local_index)
            # save the computation for the next batch
            p_y_1_to_n_minus_1 = p_y_1_to_n[best_local_index]
            # remove the chosen element from the remaining indices mask
            remaining_indices[best_local_index] = False

        # we've subset-ed our dataset twice, so we need to go back through
        # subset indices twice to recover the global indices of the chosen data
        best_local_indices = np.array(sub_pool_data.indices)[best_sub_local_indices]
        best_global_indices = np.array(pool_data.indices)[best_local_indices]
        return best_global_indices