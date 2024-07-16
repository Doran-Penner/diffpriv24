import numpy as np
import math
import privacy_accounting
import torch
from helper import device, l_inf_distances

class Aggregator:    
    """
    This is a parent class to specific aggregators
    
    ...

    Attributes
    ----------
    num_labels : int
        specifying the number of labels to be aggregated

    Methods
    -------
    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold

    """
    def __init__(self, num_labels):
        self.num_labels = num_labels

    def aggregate(self, votes):
        raise NotImplementedError

    def threshold_aggregate(self, votes, epsilon):
        raise NotImplementedError

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
    queries : list
        containing the set of q values of previous queries
    hit_max : boolean
        representing whether or not the epsilon budget is
        compeletely spent in the threshold_aggregate method

    Methods
    ----------
    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold
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
        """
        Function for aggregating teacher votes with the specified algorithm without
        passing some epsilon value, passed as a parameter to this function

        Arguments:
        :param votes: array of labels, where each label is the vote of a single 
                      teacher. so, if there are 250 teachers, the length of votes 
                      is 250
        :param epsilon: float reprepesenting the maximum epsilon that the mechanism 
                        aggregates to. this is to say, it will not report the result
                        of a vote if that would exceed the privacy budget
        :returns: integer corresponding to the aggregated label, or -1 if the response
                  would exceed the epsilon budget
        """
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
    alpha : int
        variable representing the order of the renyi divergence used in renyi differential
        privacy
    delta : float
        variable representing the delta value used for epsilon-delta differential privacy,
        specifically used when calculating the epsilon value
    prev_votes : 2-dimensional tensor 
        variable where each prev_votes[i] looks like the votes variable. needed to compare
        current votes histogram to previous ones.
    prev_labels : array 
        containing the output of each voting record in prev_votes. needed to output the 
        result of the previous votes histograms.
    gnmax : NoisyMaxAggregator instance
        used to aggregate votes when the histograms are different from previous votes
    queries : list
        containing the data dependent costs of each query with a unique response
    total_queries : int
        representing the total number of queries that have been answered
    eps_ma : float
        representing the ma epsilon for renyi differential privacy
    eprime : float
        representing the epsilon prime for renyi differential privacy
    tau_tally : int
        representing the number of times that the algorithm has responded with a previously
        given answer
    distances : function
        used to calculate the distance that is being used to compare vote histograms to
        previous vote histograms , currently we are either using the l infinity norm or we
        are using what we call the swing voter metric .
    

    Methods
    ----------
    data_dependent_cost(votes):
        function that reports the data-dependent q value, used to calculate epsilon cost

    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold
    """
    def __init__(self, scale1, scale2, p, tau, alpha=3, delta=1e-6, num_labels=10, distance_fn=l_inf_distances, epsilon_prime=privacy_accounting.epsilon_prime):
        """
        Initializer function for RepeatGNMax class

        Arguments:
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
        :param alpha: numeric representing the alpha value for the order of the renyi
                      differential privacy
        :param delta: float representing the delta needed to calculate epsilon-delta epsilons
        :param num_labels: int specifying the number of labels to be aggregated
        :param distance_fn: function that computes a distance vector to previous votes
        :param epsilon_prime: function used to calculate the epsilon prime needed for renyi
                              differential privacy . this changes with our distance function
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
        self.eprime = epsilon_prime(self.alpha, self.p, self.scale1)
        self.tau_tally = 0
        self.distances = distance_fn

    def data_dependent_cost(self,votes):
        """
        Function for calculating the data-dependent q value for a query

        Arguments:
        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
        :returns: q value
        """
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
        Function for the aggregation mechanism

        Arguments:
        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
        :returns: The label with the most votes, after adding noise to the votes to make 
                  it private.
        """
        # FIXME BAD CODE
        # we just needed it to work, but reaaally should change this
        self.total_queries += 1
        if self.prev_votes == []:
            q = self.data_dependent_cost(votes)
            self.queries.append(q)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            self.prev_votes.append(votes)
            label = self.gnmax.aggregate(votes)
            self.prev_labels.append(label)
            return label
        
        U = np.random.uniform(size=(len(votes),)) < self.p  # U is array of bools
        sub_record = votes[U]

        prev_votes = torch.tensor(np.asarray(self.prev_votes), device=device)
        divergences = self.distances(sub_record,prev_votes[:, U],self.num_labels)
        divergences += torch.normal(0, self.scale1, size=np.shape(divergences), device=device)
        min_divergence_idx = torch.argmin(divergences)

        print(divergences[min_divergence_idx])

        if divergences[min_divergence_idx] < self.tau:
            self.tau_tally += 1
            return self.prev_labels[min_divergence_idx]
        else:
            q = self.data_dependent_cost(votes)
            self.queries.append(q)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            self.prev_votes.append(votes)
            label = self.gnmax.aggregate(votes)
            self.prev_labels.append(label)
            return label

    def threshold_aggregate(self, votes, max_epsilon):
        """
        Function for aggregating teacher votes with the specified algorithm without
        passing some epsilon value, passed as a parameter to this function

        Arguments:
        :param votes: array of labels, where each label is the vote of a single 
                      teacher. so, if there are 250 teachers, the length of votes 
                      is 250
        :param epsilon: float reprepesenting the maximum epsilon that the mechanism 
                        aggregates to. this is to say, it will not report the result
                        of a vote if that would exceed the privacy budget
        :returns: integer corresponding to the aggregated label, or -1 if the response
                  would exceed the epsilon budget
        """
        # NOTE maybe we could squeeze out a couple more tau responses?
        epsilon_ma = self.eps_ma + privacy_accounting.single_epsilon_ma(
            self.data_dependent_cost(votes), self.alpha, self.scale2
        )
        ed_epsilon = privacy_accounting.renyi_to_ed(
            (self.total_queries + 1) * self.eprime + epsilon_ma,
            self.delta,
            self.alpha,
        )
        print(epsilon_ma, ed_epsilon)
        if ed_epsilon > max_epsilon:
            return -1
        return self.aggregate(votes)

class ConfidentGNMax(Aggregator):
    """
    This is a class that can aggregate teacher votes according to the algorithm described in
    the paper `Scalable Private Learning with PATE` by N. Papernot et al
    
    ...

    Attributes
    ----------
    num_labels : int
        specifying the number of labels to be aggregated
    scale1 : float
        variable affecting the amount of noise used when determining if the teachers' votes
        are confident enough to report
    scale2 : float
        variable affecting the amount of noise added to the aggregation function when 
        releasing confident votes
    tau : float 
        variable determining the confidence threshold
    alpha : int
        variable representing the order of the renyi divergence used in renyi differential
        privacy
    delta : float
        variable representing the delta value used for epsilon-delta differential privacy,
        specifically used when calculating the epsilon value
    gnmax : NoisyMaxAggregator instance
        used to aggregate votes when the votes are confident
    total_queries : int
        representing the total number of queries that have been answered
    eps_ma : float
        representing the ma epsilon for renyi differential privacy
    eprime : float
        representing the epsilon prime for renyi differential privacy
    

    Methods
    ----------
    data_dependent_cost(votes):
        function that reports the data-dependent q value, used to calculate epsilon cost

    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold
    """
    def __init__(self, scale1, scale2, tau, alpha=3, delta=1e-6, num_labels=10):
        """
        Initializer function for RepeatGNMax class

        Arguments:
        :param scale1: float variable affecting the amount of noise used when determining if 
                       the teachers' votes are confident enough to report
        :param scale2: float variable affecting the amount of noise added to the aggregation 
                       function when releasing confident votes
        :param tau: float variable determining the confidence threshold
        :param alpha: numeric representing the alpha value for the order of the renyi
                      differential privacy
        :param delta: float representing the delta needed to calculate epsilon-delta epsilons
        :param num_labels: int specifying the number of labels to be aggregated
        """
        self.scale1 = scale1
        self.scale2 = scale2
        self.tau = tau
        self.alpha = alpha
        self.delta = delta
        self.num_labels = num_labels
        self.total_queries = 0
        self.gnmax = NoisyMaxAggregator(scale2,num_labels,np.random.normal)
        self.eprime = alpha / (scale1 * scale1) 
        self.eps_ma = 0

    def data_dependent_cost(self,votes):
        """
        Function for calculating the data-dependent q value for a query

        Arguments:
        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
        :returns: q value
        """
        hist = [0]*self.num_labels
        for v in votes:
            hist[int(v)] += 1
        tot = 0
        for label in range(self.num_labels):
            if label == np.argmax(hist):
                continue
            tot += math.erfc((max(hist)-hist[label])/(2*self.scale2))
        return tot/2
 
    def aggregate(self, votes):
        """
        Function for the aggregation mechanism

        Arguments:
        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
        :returns: The label with the most votes, after adding noise to the votes to make 
                  it private.
        """
        self.total_queries += 1
        hist = torch.zeros((self.num_labels,), device=device)
        for v in votes:
            hist[v] += 1
        noised_max = torch.max(hist) + torch.normal(0, self.scale1, size=hist.shape, device=device)
        if noised_max >= self.tau:
            q = self.data_dependent_cost(votes)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            return self.gnmax.aggregate(votes)
        else:
            return -1

    def threshold_aggregate(self, votes, epsilon):
        """
        Function for aggregating teacher votes with the specified algorithm without
        passing some epsilon value, passed as a parameter to this function

        Arguments:
        :param votes: array of labels, where each label is the vote of a single 
                      teacher. so, if there are 250 teachers, the length of votes 
                      is 250
        :param epsilon: float reprepesenting the maximum epsilon that the mechanism 
                        aggregates to. this is to say, it will not report the result
                        of a vote if that would exceed the privacy budget
        :returns: integer corresponding to the aggregated label, or -1 if the response
                  would exceed the epsilon budget or if it is not confident
        """
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
