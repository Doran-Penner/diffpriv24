from aggregate import ConfidentGNMax
import datasets
import privacy_accounting
import numpy as np
import math
import torch
import globals
import scipy as sp
import cleanlab as cl

class ConfidentLearningAggregator(ConfidentGNMax):
    """
    This will be a class that will return probability vectors rather than just the 
    labels. This should not incur any extra privacy cost, so that is a good sign.    
    """

    # this is a child of ConfidentGNMax so it inherits things like data_dependent_cost
    def __init__(
            self,
            scale1,
            scale2,
            tau,
            alpha=3,
            delta=1e-6, 
            num_labels=10,
            confident=False
        ):
        # basically just the init for ConfidentGNMax but without gnmax and adding confident and hit_max
        self.scale1 = scale1
        self.scale2 = scale2
        self.tau = tau
        self.alpha = alpha
        self.delta = delta
        self.num_labels = num_labels
        self.total_queries = 0
        self.eprime = alpha / (scale1 * scale1) 
        self.eps_ma = 0
        self.confident = confident

    
    def aggregate(self, votes):
        """
        This is a GNMax or ConfidentGNMax aggregation method but it will output the 
        full probability vector rather than a label
        """

        self.total_queries += 1
        hist = torch.zeros((self.num_labels,), device=device)
        for v in votes:
            hist[v] += 1
        
        # NOTE This while even if confident is False, this will be computed, but it will not be used
        #      since we have the 'or not self.confident' a few lines below
        noised_max = torch.max(hist) + torch.normal(0, self.scale1, size=hist.shape, device=device)

        if noised_max >= self.tau or not self.confident:
            q = self.data_dependent_cost(votes)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            
            # adds noise to the histogram that we output (this is copied from NoisyRepeatMax)
            hist += self.noise_fn(loc=0.0, scale=float(self.scale), size=(self.num_labels,))

            prob_vector = sp.softmax(hist)

            return prob_vector
        else:
            return -1
        
    def threshold_aggregate(self, votes, epsilon):
        if self.confident:
            # just ConfidentGNMax.threshold_aggregate
            epsilon_ma = self.eps_ma + privacy_accounting.single_epsilon_ma(
                self.data_dependent_cost(votes), self.alpha, self.scale2
            )
            ed_epsilon = privacy_accounting.renyi_to_ed(
                (self.total_queries + 1) * self.eprime + epsilon_ma,
                self.delta,
                self.alpha,
            )
            print(epsilon_ma, ed_epsilon)
            if ed_epsilon > epsilon:
                return -1
            return self.aggregate(votes)
    
        else:
            # just NoisyMaxAggregator.threshold_aggregate
            if self.hit_max:
                return -1
            hist = [0]*self.num_labels
            for v in votes:
                hist[int(v)] += 1
            tot = 0
            for label in range(self.num_labels):
                if label == np.argmax(hist):
                    continue
                tot += math.erfc((max(hist)-hist[label])/(2*self.scale))
            if tot < 2*10e-16:
                # need a lower bound, otherwise floating-point imprecision
                # turns this into 0 and then we divide by 0
                tot = 2*10e-16
            self.queries.append(tot/2)
            eps = privacy_accounting.gnmax_epsilon(self.queries, 3, self.scale, 1e-6)
            print(eps)
            if eps > epsilon:
                print("uh oh!")
                self.hit_max = True  # FIXME this is a short & cheap solution, not the best one
                return -1
            return self.aggregate(votes)

# NOTE THESE FOLLOWING FUNCTIONS MIGHT BE IN A DIFFERENT SPOT BUT FOR THE TIME BEING THEY WILL STAY HERE
# IF YOU HAVE A PROBLEM WITH THAT I DO NOT CARE. IT IS EASIER TO KEEP TRACK OF HERE FOR NOW AND ABSTRACTION
# IS HAPPENING ELSEWHERE SO IMMA MAKE THIS FRAMEWORK BEFORE I DEAL WITH GETTING IT FORMATTED CORRECTLY

def clean_learning_PATE():
    # outline:
    # First, aggregate the teacher prediction matrix with ConfidentLearning Aggregator
    # Second, get matrix form of prediction probability vector (and calculate labels)
    # Third, clean the dataset with cl.filter()
    # Fourth, train student on cleaned data
    # Finally, return student model/get test accuracy for student

    # Things to consider
        # We need to change how the functions in get_predicted_labels.py work since we haev 
        #   an aggregator that returns a probability vector rather than just a label
        # We need to modify the student training and validation sets


    pass

def load_prediction_vectors(aggregator, dat_obj):
    """
    Basically same thing as load_predicted_labels, but returns a tuple of a matrix and an array

    :returns pred_vectors, labels:
    """


    # This is mostly the same as get_predicted_labels.load_predicted_labels, but it is slightly different
    votes = np.load(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    agg = lambda x: aggregator.threshold_aggregate(x, 10)  # noqa: E731
    pred_vectors = np.apply_along_axis(agg, 0, votes)

    # this should just take the argmin of the probability vector for a given query
    label_maker = lambda x:np.argmin(x)

    # get an array of labels
    labels = np.apply_along_axis(label_maker,0,pred_vectors)

    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy', pred_vectors)
    return pred_vectors, labels

