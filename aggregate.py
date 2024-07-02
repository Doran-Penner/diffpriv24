import numpy as np
import math
import privacy_accounting
import torch
from helper import device

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
        self.hit_max = False

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
        hist += self.noise_fn(loc=0.0, scale=float(self.scale), size=(self.num_labels,))
        label = np.argmax(hist)
        return label

    def threshold_aggregate(self,votes,epsilon):
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
        eps = privacy_accounting.gnmax_epsilon(self.queries, 2, self.scale, 0.00001)
        print(eps)
        if eps > epsilon:
            print("uh oh!")
            self.hit_max = True  # FIXME this is a short & cheap solution, not the best one
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
    def __init__(self,scale1,scale2,p,tau,alpha=3,delta=1e-5,num_labels=10):
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
        self.alpha = alpha
        self.num_labels = num_labels
        self.prev_votes = []
        self.prev_labels = []
        self.gnmax = NoisyMaxAggregator(scale2,num_labels,np.random.normal)
        self.queries = []
        self.total_queries = 0
        self.eps_ma = 0
        self.delta = delta
        self.eprime = privacy_accounting.epsilon_prime(self.alpha, self.p, self.scale1)

    def data_dependent_cost(self,votes):
        hist = [0]*self.num_labels
        for v in votes:
            hist[int(v)] += 1
        tot = 0
        for label in range(self.num_labels):
            if label == np.argmax(hist):
                continue
            tot += math.erfc((max(hist)-hist[label])/(2*self.scale2))
        return tot/2
 
    def aggregate(self,votes):
        """
        Function for the aggregation mechanism.

        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
    
        :returns: The label with the most votes, after adding noise to the votes to make 
                  it private.
        """
        # FIXME BAD CODE
        # we just needed it to work, but reaaally should change this
        if self.prev_votes == []:
            q = self.data_dependent_cost(votes)
            self.queries.append(q)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            self.prev_votes.append(votes)
            label = self.gnmax.aggregate(votes)
            self.prev_labels.append(label)
            return label
        
        self.total_queries += 1
        U = []
        for voter in range(len(votes)):
            if np.random.uniform() < self.p:
                U.append(voter)
        U = np.array(U)
        sub_record = votes[U]

        # using torch so we can do this on the gpu (for speed)
        hist = torch.zeros((self.num_labels,), device=device)
        for v in sub_record:
            hist[v] += 1
        
        prev_votes = torch.tensor(np.asarray(self.prev_votes), device=device)
        total_hist = torch.zeros((len(prev_votes), self.num_labels), device=device)

        unique, counts = torch.unique(prev_votes, dim=1, return_counts=True)
        total_hist[:,unique] = counts.float()

        total_hist += torch.normal(0, self.scale1, size=np.shape(total_hist), device=device)

        divergences, _ = torch.max(torch.abs(hist-total_hist), dim=1)
        min_divergence = torch.argmin(divergences)
        breakpoint()

        if divergences[min_divergence] < self.tau:
            return self.prev_labels[min_divergence]
        else:
            q = self.data_dependent_cost(votes)
            self.queries.append(q)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            self.prev_votes.append(votes)
            label = self.gnmax.aggregate(votes)
            self.prev_labels.append(label)
            return label

    def threshold_aggregate(self,votes,epsilon):
        thing0 = self.eps_ma + privacy_accounting.single_epsilon_ma(
                    self.data_dependent_cost(votes), self.alpha, self.scale2
                )
        print(thing0)
        if (
            privacy_accounting.renyi_to_ed(
                self.total_queries
                * self.eprime
                + thing0,
                self.delta,
                self.alpha,
            )
            > epsilon
        ):
            return -1
        return self.aggregate(votes)
