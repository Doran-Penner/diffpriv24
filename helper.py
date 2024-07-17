import torch
import numpy as np
import math
import globals

def l_inf_distances(votes, prev_votes, dat_obj):
    """
    Function used to calculate the max difference over a label between the current
    vote histogram and every previous vote histogram

    :param votes: array of labels, where each label is the vote of a single teacher. 
                  this list may be a subsample of the votes of every actual teacher.
    :param prev_votes: 2-dimensional tensor variable where each prev_votes[i] looks 
                       like the votes variable. needed to compare current votes 
                       histogram to previous ones.
    :param num_labels: int representing the number of labels in the dataset
    :returns: number (maybe technically a tensor) representing the max difference
              between vote histograms
    """
    # using torch so we can do this on the gpu (for speed)
    hist = torch.zeros((dat_obj.num_labels,), device=globals.device)
    for v in votes:
        hist[v] += 1
        
    total_hist = torch.zeros((len(prev_votes), dat_obj.num_labels), device=globals.device)

    unique, counts = torch.unique(prev_votes, dim=1, return_counts=True)
    total_hist[:,unique] = counts.float()

    divergences, _ = torch.max(torch.abs(hist-total_hist), dim=1)
    return divergences

def swing_distance(votes, prev_votes, dat_obj):
    """
    Function used to calculate the distance between two voting records using the
    swing voter distance metric . this is to say , the number of teachers that voted
    for a different label given different inputs

    :param votes: array of labels, where each label is the vote of a single teacher. 
                  this list may be a subsample of the votes of every actual teacher.
    :param prev_votes: 2-dimensional tensor variable where each prev_votes[i] looks 
                       like the votes variable. needed to compare current votes 
                       histogram to previous ones.
    :param num_labels: int representing the number of labels in the dataset
    :returns: number (maybe technically a tensor) representing the number of voters
              that changed their vote
    """
    swing_counts = torch.sum(prev_votes != torch.from_numpy(votes).to(globals.device) ,dim=1)
    return swing_counts.float()
    
def data_dependent_cost(votes, num_labels, scale2):
    """
    Function for calculating the data-dependent q value for a query

    Arguments:
    :param votes: array of labels, where each label is the vote of a single teacher. 
                  so, if there are 250 teachers, the length of votes is 250.
    :returns: q value
    """
    hist = [0]*num_labels
    for v in votes:
        hist[int(v)] += 1
    tot = 0
    for label in range(num_labels):
        if label == np.argmax(hist):
            continue
        tot += math.erfc((max(hist)-hist[label])/(2*scale2))
    return tot/2
