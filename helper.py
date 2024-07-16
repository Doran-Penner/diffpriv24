import torch
from datasets import make_dataset

# below equivalent to "pragma once" so we don't re-calculate dataset every time
if "been_run" not in vars():
    been_run = True
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("using device", device)
    
    # note: here's our single place to hard-code the dataset & num_teachers,
    # if/when we change it we should be able to just change this
    # (check that to be sure though!)
    dataset = make_dataset("svhn", 250, seed=96)

def l_inf_distances(votes, prev_votes, num_labels):
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
    hist = torch.zeros((num_labels,), device=device)
    for v in votes:
        hist[v] += 1
        
    total_hist = torch.zeros((len(prev_votes), num_labels), device=device)

    unique, counts = torch.unique(prev_votes, dim=1, return_counts=True)
    total_hist[:,unique] = counts.float()

    divergences, _ = torch.max(torch.abs(hist-total_hist), dim=1)
    return divergences

def swing_distance(votes, prev_votes, num_labels):
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
    swing_counts = torch.sum(prev_votes != torch.from_numpy(votes).to(device) ,dim=1)
    return swing_counts.float()