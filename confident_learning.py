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
        # confident scale
        self.scale1 = scale1
        # gnmax scale
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

        # used for threshold_aggregate if confident == False
        self.hit_max = False

        # used for something on GNMax's threshold_aggregate
        self.queries = []
    
    def aggregate(self, votes):
        """
        This is a GNMax or ConfidentGNMax aggregation method but it will output the 
        full probability vector rather than a label
        """

        self.total_queries += 1
        hist = torch.zeros((self.num_labels,), device=globals.device)
        for v in votes:
            hist[v] += 1
        
        noised_hist = hist + torch.normal(0, self.scale1, size=hist.shape, device=globals.device)
        noised_max = torch.max(noised_hist)

        if noised_max >= self.tau or not self.confident:
            q = self.data_dependent_cost(votes)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            
            # adds noise to the histogram that we output (this is copied from NoisyRepeatMax)
            #hist += torch.normal(0, self.scale2, size=hist.shape, device=globals.device)

            softmaxer = torch.nn.Softmax(dim=0)
            prob_vector = softmaxer(hist).to("cpu")

            return prob_vector
        else:
            return np.zeros((self.num_labels,))
        
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
                return np.zeros((self.num_labels,))
            return self.aggregate(votes)
    
        else:
            # just NoisyMaxAggregator.threshold_aggregate
            if self.hit_max:
                return np.zeros((self.num_labels,))
            hist = [0]*self.num_labels
            for v in votes:
                hist[int(v)] += 1
            tot = 0
            for label in range(self.num_labels):
                if label == np.argmax(hist):
                    continue
                tot += math.erfc((max(hist)-hist[label])/(2*self.scale2))
            if tot < 2*10e-16:
                # need a lower bound, otherwise floating-point imprecision
                # turns this into 0 and then we divide by 0
                tot = 2*10e-16
            self.queries.append(tot/2)
            eps = privacy_accounting.gnmax_epsilon(self.queries, 3, self.scale2, 1e-6)
            print(eps)
            if eps > epsilon:
                print("uh oh!")
                self.hit_max = True  # FIXME this is a short & cheap solution, not the best one
                return np.zeros((self.num_labels,))
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
    scale1 = 100
    scale2 = 100
    tau = 0.6

    dat_obj = globals.dataset
    # Change inputs to this, and make sure it doesn't use default things to be in line with other aggregators?
    agg = ConfidentLearningAggregator(scale1,scale2,tau)
    
    student_data = dat_obj.student_data
    loader = torch.utils.data.DataLoader(student_data, shuffle=False, batch_size=256)

    if not isfile(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy"):
        get_predicted_labels.calculate_prediction_matrix(loader, dat_obj)
    
    # aggregation step
    #probability_vectors, labels = load_prediction_vectors(agg,dat_obj)
    if not isfile(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy"):
        probability_vectors, labels = load_prediction_vectors(agg,dat_obj)
    else:
        # no need to recalculate if already calculated
        labels = np.load(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy")
        probability_vectors = np.load(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_probability_vectors.npy")
    
    for i in range(len(probability_vectors)):
        for j in range(len(probability_vectors[i])):
            if probability_vectors[i][j] < 0 or probability_vectors[i][j] > 1:
                print("WARNING")
                print(probability_vectors[i][j])

    # This will calculate the teacher accuracy before the cleaning happens
    # the hope is that after cleaning the data, we will have better accuracy
    # on the labeled data, even if that means there is less labeled data
    assess_teacher_aggregation(student_data,labels)

    # get the labeled indices :)
    indices_to_clean = []
    for i in range(len(labels)):
        if labels[i] != -1:
            indices_to_clean.append(i)

    # now to dump bleach onto the dataset
    # this is the most basic (ish) version of the cleanlab find_label_issues
    # there are more arguments you can have and all that, but for the time being
    # I felt like it was more important to get the framework down rather than the 
    # best possible thing in place
    # This will be an array of booleans with True at every index that we wish to 
    # unlabel
    print(len(labels[indices_to_clean]), len(indices_to_clean))
    label_issue_mask = find_label_issues(labels[indices_to_clean],probability_vectors[indices_to_clean,:],filter_by="confident_learning")
    if any(label_issue_mask):
        print("Bad label found")
    # now just use the mask we got to change the relevant labels to -1 to keep 
    # consistent with the earlier unlabeling bit
    # This is probably not the most efficient, so it will likely be changed but for now
    for i in range(len(label_issue_mask)):
        if label_issue_mask[i]:
            # since we have the indices_to_clean list, we have to make sure
            # we actually change the correct label
            labels[indices_to_clean[i]] = -1

    
    assess_teacher_aggregation(student_data,labels)
    

def load_prediction_vectors(aggregator, dat_obj):
    """
    Basically same thing as load_predicted_labels, but returns a tuple of a matrix and an array

    :returns pred_vectors, labels:
    """


    # This is mostly the same as get_predicted_labels.load_predicted_labels, but it is slightly different
    votes = np.load(f"./saved/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    agg = lambda x: aggregator.threshold_aggregate(x, 10)  # noqa: E731
    pred_vectors = np.apply_along_axis(agg, 0, votes)
    # this should just take the argmax of the probability vector for a given query
    label_maker = lambda x: np.argmax(x) if any(x) else -1

    # get an array of labels
    labels = np.apply_along_axis(label_maker,0,pred_vectors)

    pred_vectors = np.transpose(pred_vectors)
    # unsure if we need to save both the probability vectors and the labels, but just in case I will do both
    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_probability_vectors.npy', pred_vectors)
    np.save(f'./saved/{dat_obj.name}_{dat_obj.num_teachers}_agg_teacher_predictions.npy', labels)

    return pred_vectors, labels

def assess_teacher_aggregation(student_data,labels):
    """
    function to asses class-based statistics
    """
    # dict of true_label: (correct,unlabeled,total)
    by_class = {1:[0,0,0], 2:[0,0,0], 3:[0,0,0],4:[0,0,0], 5:[0,0,0],6:[0,0,0], 7:[0,0,0], 8:[0,0,0],9:[0,0,0], 0:[0,0,0]}
    total = 0
    correct = 0
    unlabeled = 0
    for i,label in enumerate(labels):
        by_class[student_data[i][1]][2] += 1
        total += 1
        if label == -1:
            unlabeled += 1
            by_class[student_data[i][1]][1] += 1
        elif label == student_data[i][1]:
            by_class[student_data[i][1]][0] += 1
            correct += 1

    labeled = total-unlabeled

    print("Assessment Results:")
    print("\t\t\tlabeled:\tcorrect:\ttotal:\t\tlabel_accuracy:")
    print(f"Overall Summary:\t{labeled}\t\t{correct}\t\t{total}\t\t{correct/labeled}")
    for key in by_class:
        print(f"Summary of {key}:\t\t{by_class[key][2]-by_class[key][1]}\t\t{by_class[key][0]}\t\t{by_class[key][2]}\t\t{by_class[key][0]/(by_class[key][2]-by_class[key][1])}")

        
                







if __name__ == "__main__":
    clean_learning_PATE()