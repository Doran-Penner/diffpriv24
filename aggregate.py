import numpy as np
import torch


def lnmax(votes, scale, num_labels = 10):
    """
    Function for aggregating teacher votes according to the algorithm described
    in the original PATE paper. This function is essentially ReportNoisyMax with
    Laplacian noise.

    Arguments:
    votes -------- array of labels, where each label is the vote of a single 
                   teacher. so, if there are 250 teachers, the length of votes 
                   is 250.
    scale -------- variable affecting the amount of noise. The larger the scale 
                   value, the noisier it is. ReportNoisyMax is epsilon 
                   differentially private if scale is equal to 1/epsilon.
    num_labels --- number of possible labels that the teachers can vote for. so,
                   for the MNIST dataset, num_labels is equal to 10.

    Outputs: The label with the most votes, after adding noise to the votes to
             make it private.
    """
    hist = [0]*num_labels
    for v in votes:
        hist[v] += 1
    for label in range(num_labels):
        hist[label] += np.random.laplace(loc=0.0,scale=float(scale))
    label = np.argmax(hist)
    return label

def repeat_gnmax(votes, scale1, scale2, p, tau, prev_votes, prev_labels, num_labels):
    """
    Function for aggregating teacher votes according to the algorithm that Tory
    developed, called Repeat-GNMax.

    Arguments:
    votes -------- array of labels, where each label is the vote of a single 
                   teacher. so, if there are 250 teachers, the length of votes 
                   is 250.
    scale1 ------- numeric variable affecting the amount of noise when comparing the
                   current voting record to the older voting records.
    scale2 ------- numeric variable affecting the amount of noise added to the
                   aggregation function when releasing the results of queries that 
                   don't have similar previous queries.
    p ------------ numeric variable affecting the poisson sampling. each teacher has
                   probability p of being included in the sample.
    tau ---------- numeric variable determining the threshold of similarity that the
                   vote histograms have to be to release the same answer. so, the
                   lower the threshold, the more similar the histograms need to be.
    prev_votes --- 2-dimensional tensor where each prev_votes[i] looks like the votes
                   variable. needed to compare curret votes histogram to previous ones.
    prev_labels -- array containing the output of each voting record in prev_votes. 
                   needed to output the result of the previous votes histograms.
    num_labels --- number of possible labels that the teachers can vote for. so,
                   for the MNIST dataset, num_labels is equal to 10.
    
    Outputs: The label with the most votes, after adding noise to the votes to
             make it private.
    """
    U = []
    for voter in range(len(votes)):
        if np.random.uniform() < p:
            U.append(voter)
    U = np.array(U)
    sub_record = votes[U]
    hist = [0]*num_labels
    for v in sub_record:
        hist[v] += 1
    seen = False
    which_record = 0
    for record in prev_votes:
        new_hist = [0]*num_labels
        for v in record:
            new_hist[v] += 1
        for label in range(num_labels):
            hist[label] += np.random.gaussian(loc=0.0,scale=float(scale1))
        divergence = np.max(np.abs(hist-new_hist))
        if divergence < tau:
            seen = True
            break
        which_record += 1
    if seen:
        return prev_labels[which_record]
    else:
      return lnmax(votes, scale2)
