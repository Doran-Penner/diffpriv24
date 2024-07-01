import numpy as np
import math
from privacy_accounting import repeat_epsilon, gnmax_epsilon

class Aggregator:    
    """
    This is a parent class to specific aggregators
    
    ...

    Attributes
    ----------
    num_labels : int
        specifying the number of labels to be aggregated

    Methods
    ----------
    aggregate(votes)
        function that returns the result of the aggregation mechanism
    """
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def aggregate(self, votes):
        return 0

    def threshold_aggregate(self, votes, epsilon):
        return 0

class NoisyMaxAggregator(Aggregator):
    """
    This is a general class that can do ReportNoisyMax with laplacian noise or 
    with gaussian noise.
    
    ...

    Attributes
    ----------
    num_labels : int
        specifying the number of labels to be aggregated
    scale : float
        specifying the amount of noise. The larger the scale 
        value, the noisier it is. ReportNoisyMax is epsilon 
        differentially private if scale is equal to 1/epsilon
    noise_fn : function
        specifying the distribution that the noise must
        be drawn from. for basic ReportNoisyMax, this is the
        Laplacian distribution

    Methods
    ----------
    aggregate(votes)
        function that returns the result of the aggregation mechanism
    """
    def __init__(self, scale, num_labels=10, noise_fn=np.random.laplace):
        """
        Initializer function for NoisyMaxAggregator class
        :param scale: float specifying the amount of noise. The larger the scale 
                      value, the noisier it is. ReportNoisyMax is epsilon 
                      differentially private if scale is equal to 1/epsilon
        :param num_labels: int specifying the number of labels that the teacher
                           can vote for. so, for the MNIST dataset, num_labels
                           is equal to 10
        :param noise_fn: function specifying the distribution that the noise must
                         be drawn from. for basic ReportNoisyMax, this is the
                         Laplacian distribution
        """
        self.scale = scale
        self.num_labels = num_labels
        self.noise_fn=noise_fn
        self.queries = []

    def aggregate(self,votes):
        """
        Function for aggregating teacher votes according to the algorithm described
        in the original PATE paper. This function is essentially ReportNoisyMax with
        Laplacian noise.

        Arguments:
        :param votes: array of labels, where each label is the vote of a single 
                       teacher. so, if there are 250 teachers, the length of votes 
                       is 250
        :return: index indicating the max argument in the array passed to the function
        """
        hist = [0]*self.num_labels
        for v in votes:
            hist[int(v)] += 1
        for label in range(self.num_labels):
            hist[label] += self.noise_fn(loc=0.0,scale=float(self.scale))
        label = np.argmax(hist)
        return label

    def threshold_aggregate(self,votes,epsilon):
        hist = [0]*self.num_labels
        for v in votes:
            hist[int(v)] += 1
        tot = 0
        for label in range(self.num_labels):
            if label == np.argmax(hist):
                continue
            tot += math.erfc(max(hist)-hist[label]/(2*self.scale))
        if tot < 2*10e-16:
            tot = 2*10e-16
        self.queries.append(tot/2)
        eps = gnmax_epsilon(self.queries, 2, self.scale, 0.00001)
        print(eps)
        if eps > epsilon:
            print("uh oh!")
            return -1
        return self.aggregate(votes)

class RepeatGNMax(Aggregator):
    """
    This is a class that can aggregate teacher votes according to the algorithm that
    Tory developed, called Repeat-GNMax.
    
    ...

    Attributes
    ----------
    num_labels : int
        specifying the number of labels to be aggregated
    scale1 : float
        variable affecting the amount of noise when comparing the 
        current voting record to the older voting records.
    scale2 : float
        variable affecting the amount of noise added to the aggregation function when 
        releasing the results of queries that don't have similar previous queries.
    p : float
        variable affecting the poisson sampling. each teacher hasprobability p of
        being included in the sample.
    tau : float 
        variable determining the threshold of similarity that the vote histograms have 
        to be to release the same answer. so, the lower the threshold, the more similar
        the histograms need to be.
    prev_votes : 2-dimensional tensor 
        variable where each prev_votes[i] looks like the votes variable. needed to compare
        current votes histogram to previous ones.
    prev_labels : array 
        containing the output of each voting record in prev_votes. needed to output the 
        result of the previous votes histograms.
    gnmax : NoisyMaxAggregator instance
        used to aggregate votes when the histograms are different from previous votes

    Methods
    ----------
    aggregate(votes)
        function that returns the result of the aggregation mechanism
    """
    def __init__(self,scale1,scale2,p,tau,num_labels=10):
        """
        Initializer function for RepeatGNMax class
        :param num_labels: int specifying the number of labels to be aggregated
        :param scale1: float variable affecting the amount of noise when comparing the 
                       current voting record to the older voting records.
        :param scale2: float variable affecting the amount of noise added to the 
                       aggregation function when releasing the results of queries that 
                       don't have similar previous queries.
        :param p: float variable affecting the poisson sampling. each teacher hasprobability 
                  p of being included in the sample.
        :param tau: float variable determining the threshold of similarity that the vote 
                    histograms have to be to release the same answer. so, the lower the 
                    threshold, the more similar the histograms need to be.
        """
        self.scale1 = scale1
        self.scale2 = scale2
        self.p = p
        self.tau = tau
        self.num_labels = num_labels
        self.prev_votes = []
        self.prev_labels = []
        self.gnmax = NoisyMaxAggregator(scale2,num_labels,np.random.gaussian)
        self.queries = []
        self.total_queries = 0

    def data_dependant_cost(self,votes):
        hist = [0]*self.num_labels
        for v in votes:
            hist[int(v)] += 1
        tot = 0
        for label in range(self.num_labels):
            if label == np.argmax(hist):
                continue
            tot += math.erfc(max(hist)-hist[label]/(2*self.scale))
        return tot/2
 
    def aggregate(self,votes):
        """
        Function for the aggregation mechanism.

        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
    
        :returns: The label with the most votes, after adding noise to the votes to make 
                  it private.
        """
        self.total_queries += 1
        U = []
        for voter in range(len(votes)):
            if np.random.uniform() < self.p:
                U.append(voter)
        U = np.array(U)
        sub_record = votes[U]
        hist = [0]*self.num_labels
        for v in sub_record:
            hist[v] += 1
        seen = False
        which_record = 0
        for record in self.prev_votes:
            new_hist = [0]*self.num_labels
            for v in U:
                new_hist[record[v]] += 1
            for label in range(self.num_labels):
                hist[label] += np.random.gaussian(loc=0.0,scale=float(self.scale1))
            divergence = np.max(np.abs(hist-new_hist))
            if divergence < self.tau:
                seen = True
                break
            which_record += 1
        if seen:
            return self.prev_labels[which_record]
        else:
            self.queries.append(self.data_dependant_cost(votes))
            self.prev_votes.append(votes)
            label = self.gnmax.aggregate(votes)
            self.prev_labels.append(label)
            return label

    def threshold_aggregate(self,votes,epsilon):
        if repeat_epsilon(self.queries + [self.data_dependant_cost(votes)], self.total_queries, 2, self.scale1, self.scale2, self.p, 0.00001) > epsilon:
            return -1
        return self.aggregate(votes)
