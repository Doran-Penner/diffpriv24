from aggregate import ConfidentGNMax
import datasets
import privacy_accounting
import numpy as np
import math
import torch
import globals
import scipy as sp
from cleanlab.filter import find_label_issues
import get_predicted_labels
from os.path import isfile


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
        # At this moment, I don't know what adding -1 labels will do to the confident learning
        # bit, so I want to keep confident as false for the time being until we figure out how 
        # to properly deal with that in an easy way
        self.confident = confident

    
    def aggregate(self, votes):
        """
        This is a GNMax or ConfidentGNMax aggregation method but it will output the 
        full probability vector rather than a label
        """

        self.total_queries += 1
        hist = torch.zeros((self.num_labels,), device=globals.device)
        for v in votes:
            hist[v] += 1
        
        # NOTE This while even if confident is False, this will be computed, but it will not be used
        #      since we have the 'or not self.confident' a few lines below
        noised_max = torch.max(hist) + torch.normal(0, self.scale1, size=hist.shape, device=globals.device)

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

    # possible values for the aggregator, they are dummy variables for now.
    scale1 = 10
    scale2 = 10
    tau = 0.6

    dat_obj = globals.dataset
    # Change inputs to this, and make sure it doesn't use default things to be in line with other aggregators?
    agg = ConfidentLearningAggregator(scale1,scale2,tau)
    
    

    if not isfile(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy"):
        student_data = dat_obj.student_data
        loader = torch.utils.data.DataLoader(student_data, shuffle=False, batch_size=256)
        get_predicted_labels.calculate_prediction_matrix(loader, dat_obj)
    
    # aggregation step
    probability_vectors, labels = load_prediction_vectors(agg,dat_obj)

    # This will calculate the teacher accuracy before the cleaning happens
    # the hope is that after cleaning the data, we will have better accuracy
    # on the labeled data, even if that means there is less labeled data
    correct = 0
    guessed = 0
    unlabeled = 0
    for i, label in enumerate(labels):
        guessed += 1
        if label == student_data[i][1]:
            correct += 1
        if label == -1:
            unlabeled += 1
    labeled = guessed-unlabeled
    


    # now to dump bleach onto the dataset
    # this is the most basic (ish) version of the cleanlab find_label_issues
    # there are more arguments you can have and all that, but for the time being
    # I felt like it was more important to get the framework down rather than the 
    # best possible thing in place
    # This will be an array of booleans with True at every index that we wish to 
    # unlabel
    if not agg.confident:
        # I don't have a great idea of how -1s would affect the find_label_issues
        # function, so I am going to call find_label_issues on just the bits that
        # come before we go over the epsilon value (labels != -1)
        label_issue_mask = find_label_issues(labels[:labeled],probability_vectors[:labeled],filter_by="both")
    else:
        print("WARNING, ENTERING UNKOWN TERRITORY")
        label_issue_mask = find_label_issues(labels,probability_vectors,filter_by="both")

    # now just use the mask we got to change the relevant labels to -1 to keep 
    # consistent with the earlier unlabeling bit
    # This is probably not the most efficient, so it will likely be changed but for now
    for i in range(len(label_issue_mask)):
        if label_issue_mask[i]:
            labels[i] = -1

    new_correct = 0
    new_unlabeled = 0
    for i, label in enumerate(labels):
        if label == student_data[i][1]:
            new_correct += 1
        if label == -1:
            new_unlabeled += 1
    new_labeled = guessed - new_unlabeled

    print("Summary time!!!")
    print()
    print("\t\tlabeled:\tlabel_acc\ttotal_acc")
    print(f"Pre Cleaning:\t{labeled}\t\t{correct/labeled}\t\t{correct/guessed}")
    print(f"Post Cleaning:\t{new_labeled}\t\t{new_correct/new_labeled}\t\t{new_correct/guessed}")
    

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


    # unsure if we need to save both the probability vectors and the labels, but just in case I will do both
    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_probability_vectors.npy', pred_vectors)
    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy', labels)
    return pred_vectors, labels

