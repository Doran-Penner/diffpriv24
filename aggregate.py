import numpy as np
import torch

def aggregate(votes, scale):
    hist = [0]*10
    for v in votes:
        hist[v] += 1
    for label in range(10):
        hist[label] += np.random.laplace(loc=0.0,scale=float(scale))
    label = np.argmax(hist)
    return label

def repeat_gnmax(votes, scale1, scale2, p, tau, prev_votes, prev_labels)
    U = []
    for voter in range(len(votes)):
        if np.random.uniform() < p:
            U.append(voter)
    U = np.array(U)
    sub_record = votes[U]
    hist = [0]*10
    for v in sub_record:
        hist[v] += 1
    seen = False
    which_record = 0
    for record in prev_votes:
        new_hist = [0]*10
        for v in record:
            new_hist[v] += 1
        for label in range(10):
            hist[label] += np.random.gaussian(loc=0.0,scale=float(scale1))
        divergence = np.max(np.abs(hist-new_hist))
        if divergence < tau:
            seen = True
            break
        which_record += 1
    if seen_before:
        return prev_labels[which_record]
    else:
      return aggregate(votes, scale2)
