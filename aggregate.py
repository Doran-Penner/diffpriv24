import numpy as np
import math
import privacy_accounting
import torch
from helper import swing_distance, data_dependent_cost
import globals

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
    def __init__(self, dat_obj):
        self.num_labels = dat_obj.num_labels

    def aggregate(self, votes):
        """
        Function for the aggregation mechanism

        Arguments:
        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
        :returns: The label with the most votes, after applying the relevant aggregation
                  mechanism
        """
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
    def __init__(self, scale, dat_obj, noise_fn=np.random.laplace, alpha_set=list(range(2,11))):
        """
        Initializer function for NoisyMaxAggregator class
        :param scale: float specifying the amount of noise. The larger the scale 
                      value, the noisier it is. ReportNoisyMax is epsilon 
                      differentially private if scale is equal to 1/epsilon
        :param dat_obj: datasets._Dataset object representing the dataset that is being
                        aggregated over. used to find self.num_labels
        :param noise_fn: function specifying the distribution that the noise must
                         be drawn from. for basic ReportNoisyMax, this is the
                         Laplacian distribution
        """
        self.scale = scale
        self.num_labels = dat_obj.num_labels
        self.noise_fn=noise_fn
        self.queries = []
        self.hit_max = False
        # NOTE: maybe better way to do this
        self.alpha = 2
        self.alpha_set = alpha_set
        self.eps = 0

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

    def threshold_aggregate(self, votes, max_epsilon):
        """
        Function for aggregating teacher votes with the specified algorithm without
        passing some epsilon value, passed as a parameter to this function

        Arguments:
        :param votes: array of labels, where each label is the vote of a single 
                      teacher. so, if there are 250 teachers, the length of votes 
                      is 250
        :param max_epsilon: float reprepesenting the maximum epsilon that the mechanism 
                        aggregates to. this is to say, it will not report the result
                        of a vote if that would exceed the privacy budget
        :returns: integer corresponding to the aggregated label, or -1 if the response
                  would exceed the epsilon budget
        """
        if self.hit_max:
            return -1
        data_dep = data_dependent_cost(votes, self.num_labels, self.scale)
        self.queries.append(data_dep)
        eps = self.best_eps(self.queries, self.scale, max_epsilon, 1e-6)
        self.eps = eps  # for keeping state
        print(eps)
        if eps > max_epsilon:
            print("uh oh!")
            self.hit_max = True
            return -1
        return self.aggregate(votes)
    
    def best_eps(self, qs, scale, max_epsilon, delta):
        """
        todo document: this is for optimizing alpha
        """
        # (this is a bit cursed and could just be a loop, but it made sense in the moment)
        eps = privacy_accounting.gnmax_epsilon(self.queries, self.alpha, self.scale, 1e-6)
        if eps <= max_epsilon or len(self.alpha_set) <= 1:
            return eps
        else:
            self.alpha_set.remove(self.alpha)
            self.alpha = self.alpha_set[0]
            return self.best_eps(qs, scale, max_epsilon, delta)


class RepeatAggregator(Aggregator):
    """
    todo document better
    abstract aggregator that aggregates based on past votes
    NOTE: tau is either None or a tau value, if none then we always return closest
    """

    def __init__(self, noise_scale, tau, dat_obj, distance_fn, accountant, p=1.0, delta=1e-6, prev_votes=[], prev_labels=[]):
        self.noise_scale = noise_scale
        self.num_labels = dat_obj.num_labels
        self.num_teachers = dat_obj.num_teachers
        self.prev_votes = prev_votes
        self.prev_labels = prev_labels
        self.distance_fn = distance_fn
        self.accountant = accountant
        self.delta = delta
        self.require_confident = tau is not None
        self.tau = tau
        self.p = p
    
    def aggregate(self, votes):

        if self.prev_votes == []:
            print("UH-OH: no prev_votes!")
            return -1

        dist_vec = [self.distance_fn(votes, prev_vote) for prev_vote in self.prev_votes]

        shortest_dist = calc_shortest_dist()
        nearest_label = func_of(shortest_dist)

        if not self.require_confident or shortest_dist < self.tau:
            return nearest_label
        else:
            return -1

    def threshold_aggregate(self, votes, max_epsilon):
        pass

    def add_vote(self, votes, label):
        self.prev_votes.append(votes)
        self.prev_labels.append(label)


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
    dat_obj : datasets._Dataset object
        representing the dataset that we are aggregating over

    Methods
    ----------
    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold
    """
    def __init__(self, scale1, scale2, p, tau, dat_obj, distance_fn, alpha=3, delta=1e-6, epsilon_prime=privacy_accounting.epsilon_prime):
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
        :param dat_obj: datasets._Dataset variable representing the dataset that we are
                        aggregating over
        :param distance_fn: function that computes a distance vector to previous votes
        :param alpha: numeric representing the alpha value for the order of the renyi
                      differential privacy
        :param delta: float representing the delta needed to calculate epsilon-delta epsilons
        :param epsilon_prime: function used to calculate the epsilon prime needed for renyi
                              differential privacy . this changes with our distance function
        """
        self.scale1 = scale1
        self.scale2 = scale2
        self.p = p
        self.tau = tau
        self.alpha = alpha
        self.dat_obj = dat_obj
        self.num_labels = dat_obj.num_labels
        self.prev_votes = []
        self.prev_labels = []
        self.gnmax = NoisyMaxAggregator(scale2,dat_obj,np.random.normal)
        self.queries = []
        self.total_queries = 0
        self.eps_ma = 0
        self.delta = delta
        self.eprime = epsilon_prime(self.alpha, self.p, self.scale1)
        self.tau_tally = 0
        self.distances = distance_fn

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
            q = data_dependent_cost(votes, self.num_labels, self.scale2)
            self.queries.append(q)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.scale2)
            self.prev_votes.append(votes)
            label = self.gnmax.aggregate(votes)
            self.prev_labels.append(label)
            return label
        
        U = np.random.uniform(size=(len(votes),)) < self.p  # U is array of bools
        sub_record = votes[U]

        prev_votes = torch.tensor(np.asarray(self.prev_votes), device=globals.device)
        divergences = self.distances(sub_record,prev_votes[:, U],self.dat_obj)
        divergences += torch.normal(
            0, self.scale1 * math.sqrt(len(self.queries) / 2),
            size=np.shape(divergences), device=globals.device,
        )
        min_divergence_idx = torch.argmin(divergences)

        print(divergences[min_divergence_idx])

        if divergences[min_divergence_idx] < self.tau:
            self.tau_tally += 1
            return self.prev_labels[min_divergence_idx]
        else:
            q = data_dependent_cost(votes, self.num_labels, self.scale2)
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
        :param max_epsilon: float reprepesenting the maximum epsilon that the mechanism 
                            aggregates to. this is to say, it will not report the result
                            of a vote if that would exceed the privacy budget
        :returns: integer corresponding to the aggregated label, or -1 if the response
                  would exceed the epsilon budget
        """
        # NOTE maybe we could squeeze out a couple more tau responses?
        epsilon_ma = self.eps_ma + privacy_accounting.single_epsilon_ma(
            data_dependent_cost(votes, self.num_labels, self.scale2), self.alpha, self.scale2
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
    data_dependent_cost(votes, num_labels, scale2):
        function that reports the data-dependent q value, used to calculate epsilon cost

    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold
    """
    def __init__(self, scale1, scale2, tau, dat_obj, alpha=3, delta=1e-6):
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
        self.num_labels = dat_obj.num_labels
        self.total_queries = 0
        self.gnmax = NoisyMaxAggregator(scale2,dat_obj,np.random.normal)
        self.eprime = alpha / (scale1 * scale1) 
        self.eps_ma = 0

 
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
        hist = torch.zeros((self.num_labels,), device=globals.device)
        for v in votes:
            hist[v] += 1
        noised_max = torch.max(hist) + torch.normal(0, self.scale1, size=hist.shape, device=globals.device)
        if noised_max >= self.tau:
            q = data_dependent_cost(votes, self.num_labels, self.scale2)
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
            data_dependent_cost(votes, self.num_labels, self.scale2), self.alpha, self.scale2
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

class PartRepeatGNMax(Aggregator):
    """ 
    A modified RepeatGNMax aggregator that will do some number of queries using a GNMax aggregator
    And then it will do the remaining queries with a laplacian report noisy max repeat mechanism
    

    Attributes
    ----------
    num_labels : int
        size of the label space (number of possible labels in the dataset)
    GNMax_scale : float
        variable affecting the amount of noise added to the aggregation function when 
        releasing initial GNMax labels
    p : float
        variable affecting the poisson sampling. each teacher has probability p of
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
    queries : list
        containing the data dependent costs of each query with a unique response
    total_queries : int
        representing the total number of queries that have been answered
    eps_ma : float
        representing the moments accountant epsilon for renyi differential privacy
    delta : float
        representing the delta in epsilon-delta differential privacy calculations
    tau_tally : int
        representing the number of times that the algorithm has responded with a previously
        given answer
    max_num : int
        representing the number of data points to be labeled by GNMax before switching to
        Lapacian repeat
    confident : boolean
        represents whether or not we use the tau value to 
    distance_fn : function
        represents the function with which to compare a given vote to older votes
    eprime : float
        represents the epsilon value for the laplacian repeat mechanism (named for 
        consistency)
    lap_scale : int
        represents the scale that we divide the epsilon threshold by 

    Methods
    ----------
    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold
    """

    def __init__(
            self,
            GNMax_scale,
            p,
            tau,
            dat_obj,
            delta=1e-6,
            distance_fn = swing_distance,
            max_num = 1000,
            confident = True,
            eprime = privacy_accounting.laplacian_eps_prime,
            lap_scale = 50,
            GNMax_epsilon = 5,
            alpha_set = list(range(2,11)),
        ):

        # general attributes
        self.num_labels = dat_obj.num_labels
        self.total_queries = 0
        self.queries = []

        # GNMax attributes
        self.GNMax_scale = GNMax_scale
        self.gnmax = NoisyMaxAggregator(GNMax_scale,dat_obj,np.random.normal, alpha_set=alpha_set.copy())
        self.alpha = 2
        self.alpha_set = alpha_set
        self.max_num= max_num
        
        self.on_first_part = True

        # Repeat attributes
        self.p = p
        self.tau = tau
        self.prev_votes = []
        self.prev_labels = []
        self.distance_fn = distance_fn
        self.tau_tally = 0       
        self.confident = confident # possibly useful possibly not, unsure
        self.GNMax_epsilon = GNMax_epsilon
        self.lap_scale = lap_scale


        # privacy attributes
        self.eps_ma = 0 # RDP epsilon for moments accountant stuff
        self.delta = delta
        self.ed_epsilon = 0 # overall epsilon
        self.gn_epsilon = 0 # the epsilon used by GNMax
        self.ed_delta = 0 # overall delta, can this just be the other delta?
        self.eprime, self.dprime = eprime(p,2/lap_scale) # how to calculate the scale for the laplace noise?

        # Things to optimize(?):
        # use of confident?
        # what is a good lap scale
        # what is a good GNMax scale?
        # other things?
    


    def aggregate(self, votes):
        """
        Function for the aggregation mechanism. First we answer self.max_num number of
        GNMax queries, and then switch to comparing that to a lapacian report noisy min
        comparison to the GNMax votes

        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
    
        :returns: The label with the most votes, after adding noise to the votes to make 
                  it private.
        """
        
        self.total_queries += 1

        # Create an array of boolean values for the poisson sub_sampling
        sub_samp = np.random.uniform(size=(len(votes),)) < self.p
        sub_record = votes[sub_samp]

        # tensor-ize it for efficiency (?)
        prev_votes = torch.tensor(np.asarray(self.prev_votes), device=globals.device)

        # take the same sub_sample of each of the previous records, and compute the distance
        # away from the current voting record, add laplacian noise, then 
        divergences = self.distance_fn(sub_record,prev_votes[:, sub_samp],self.num_labels)
        divergences += torch.distributions.laplace.Laplace(
            # all of this is to make the distribution be created on the device
            loc=torch.zeros((), device=globals.device),
            scale=torch.full((), self.lap_scale, device=globals.device),
        ).sample(divergences.shape)
        min_divergence_idx = torch.argmin(divergences)

        # everything after this should be post-processing since we have report noisy min above this
        if divergences[min_divergence_idx] < self.tau or not self.confident:
            self.tau_tally += 1
            return self.prev_labels[min_divergence_idx]
        else:
            return -1
        
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
        
        # NOTE this assumes a few things:
        # 1) renyi composition is better than strong composition
        # 2) strong composition of a renyi composition and another strong composition is
        #    still more optimal than a composition of a renyi composition and a bunch of 
        #    individual mechanisms

        if self.on_first_part:
            # do gnmax up to the threshold
            label = self.gnmax.threshold_aggregate(votes, self.GNMax_epsilon)
            self.ed_epsilon = self.gnmax.eps

            hit_limit = label == -1 or self.total_queries >= (self.max_num - 1) or self.ed_epsilon >= self.GNMax_epsilon

            if not hit_limit:
                self.prev_votes.append(votes)
                self.prev_labels.append(label)
                self.total_queries += 1
                self.gn_epsilon = self.ed_epsilon = self.gnmax.eps
                return label
            else:
                self.on_first_part = False
                # recompute over all alphas to find best final alpha
                costs = [
                    privacy_accounting.renyi_to_ed(
                        privacy_accounting.epsilon_ma_vec(
                            self.gnmax.queries, alpha, self.GNMax_scale
                        ),
                        self.delta,
                        alpha,
                    )
                    for alpha in self.alpha_set
                ]
                min_cost = min(costs)
                print("MIN FINAL COST:", min_cost)
                min_cost_index = costs.index(min_cost)  # argmin(costs)
                self.alpha = self.alpha_set[min_cost_index]
                self.gn_epsilon = min_cost
                # re-start, now on second partition
                return self.threshold_aggregate(votes, max_epsilon)
        else:
            # do rep_lnmax
            # need a delta_prime, but we can just default it to 1e-6
            temp_epsilon, temp_delta = privacy_accounting.homogeneous_strong_composition(
                    self.eprime,
                    0,
                    1e-6,
                    self.tau_tally + 1
                )

            # now we can just combine the two other composed things? i think?
            self.ed_epsilon, self.ed_delta = (
                privacy_accounting.heterogeneous_strong_composition(
                    np.asarray([self.gn_epsilon, temp_epsilon]),
                    np.asarray([self.delta, temp_delta]),
                    1e-6,
                )
            )
            
            print(temp_epsilon, self.ed_epsilon)
            if self.ed_epsilon > max_epsilon:
                return -1
            else:
                return self.aggregate(votes)


class LapRepeatGNMax(Aggregator):
    """ 
    This is a RepeatGNMax-like aggregator that uses laplacian report noisy min for the
    previous votes comparison.
    

    Attributes
    ----------
    num_labels : int
        size of the label space (number of possible labels in the dataset)
    GNMax_scale : float
        variable affecting the amount of noise added to the aggregation function when 
        releasing initial GNMax labels
    p : float
        variable affecting the poisson sampling. each teacher has probability p of
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
    queries : list
        containing the data dependent costs of each query with a unique response
    total_queries : int
        representing the total number of queries that have been answered
    eps_ma : float
        representing the moments accountant epsilon for renyi differential privacy
    delta : float
        representing the delta in epsilon-delta differential privacy calculations
    tau_tally : int
        representing the number of times that the algorithm has responded with a previously
        given answer
    max_num : int
        representing the number of data points to be labeled by GNMax before switching to
        Lapacian repeat
    confident : boolean
        represents whether or not we use the tau value to 
    distance_fn : function
        represents the function with which to compare a given vote to older votes
    eprime : float
        represents the epsilon value for the laplacian repeat mechanism (named for 
        consistency)
    lap_scale : int
        represents the scale that we divide the epsilon threshold by 

    Methods
    ----------
    aggregate(votes):
        function that returns the result of the aggregation mechanism

    treshold_aggregate(votes, epsilon):
        function that aggregates votes until the epsilon spent reaches a certain threshold
    """

    def __init__(
            self,
            GNMax_scale,
            lap_scale,
            p,
            tau,
            dat_obj,
            alpha=3,
            delta=1e-6,
            distance_fn = swing_distance,
            eprime = privacy_accounting.laplacian_eps_prime,
        ):

        # general attributes
        self.num_labels = dat_obj.num_labels
        self.total_queries = 0
        self.queries = []

        # GNMax attributes
        self.GNMax_scale = GNMax_scale
        self.gnmax = NoisyMaxAggregator(GNMax_scale,dat_obj,np.random.normal)
        self.alpha = alpha

        # Repeat attributes
        self.p = p
        self.tau = tau
        self.prev_votes = []
        self.prev_labels = []
        self.distance_fn = distance_fn
        self.tau_tally = 0       
        self.lap_scale = lap_scale


        # privacy attributes
        self.eps_ma = 0 # RDP epsilon for moments accountant stuff
        self.delta = delta
        self.ed_epsilon = 0 # overall epsilon
        self.ed_delta = 0 # overall delta, can this just be the other delta?
        self.gn_epsilon = 0
        self.eprime, self.dprime = eprime(p,2/lap_scale) # how to calculate the scale for the laplace noise?

        # Things to optimize(?):
        # use of confident?
        # what is a good lap scale
        # what is a good GNMax scale?
        # other things?
    


    def aggregate(self, votes):
        """
        Function for the aggregation mechanism. First we answer self.max_num number of
        GNMax queries, and then switch to comparing that to a lapacian report noisy min
        comparison to the GNMax votes

        :param votes: array of labels, where each label is the vote of a single teacher. 
                      so, if there are 250 teachers, the length of votes is 250.
    
        :returns: The label with the most votes, after adding noise to the votes to make 
                  it private.
        """
        
        self.total_queries += 1

        # check if we are still below the threshold for number of GNMax queries
        if len(self.prev_votes) == 0:
            # calculate and store epsilon cost for this query
            q = data_dependent_cost(votes, self.num_labels, self.GNMax_scale)
            self.queries.append(q)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.GNMax_scale)

            # store the teacher responses to this query for later reference
            self.prev_votes.append(votes)

            # choose the best label with GNMax and save that value
            label = self.gnmax.aggregate(votes)
            self.prev_labels.append(label)

            return label
        
        # otherwise, do Lapacian Repeat Mechanism

        # Create an array of boolean values for the poisson sub_sampling
        sub_samp = np.random.uniform(size=(len(votes),)) < self.p
        sub_record = votes[sub_samp]

        # tensor-ize it for efficiency (?)
        prev_votes = torch.tensor(np.asarray(self.prev_votes), device=globals.device)

        # take the same sub_sample of each of the previous records, and compute the distance
        # away from the current voting record, add laplacian noise, then 
        divergences = self.distance_fn(sub_record,prev_votes[:, sub_samp],self.num_labels)
        divergences += torch.distributions.laplace.Laplace(
            # all of this is to make the distribution be created on the device
            loc=torch.zeros((), device=globals.device),
            scale=torch.full((), self.lap_scale, device=globals.device),
        ).sample(divergences.shape)
        min_divergence_idx = torch.argmin(divergences)

        # everything after this should be post-processing since we have report noisy min above this
        if divergences[min_divergence_idx] < self.tau or not self.confident:
            self.tau_tally += 1
            return self.prev_labels[min_divergence_idx]
        else:
            # calculate and store epsilon cost for this query
            q = data_dependent_cost(votes, self.num_labels, self.GNMax_scale)
            self.queries.append(q)
            self.eps_ma += privacy_accounting.single_epsilon_ma(q, self.alpha, self.GNMax_scale)

            # store the teacher responses to this query for later reference
            self.prev_votes.append(votes)

            # choose the best label with GNMax and save that value
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
        
        # NOTE this assumes a few things:
        # 1) renyi composition is better than strong composition
        # 2) strong composition of a renyi composition and another strong composition is
        #    still more optimal than a composition of a renyi composition and a bunch of 
        #    individual mechanisms

        # NOTE maybe we could squeeze out a couple more tau responses?
        # here is the hypothetical data dependent cost of gnmaxing the vote vector
        # we assume that this will be vector will not be repeat labeled
        temp_epsilon_ma = self.eps_ma + privacy_accounting.single_epsilon_ma(
                data_dependent_cost(votes, self.num_labels, self.GNMax_scale), self.alpha, self.GNMax_scale
            )
        self.gn_epsilon = privacy_accounting.renyi_to_ed(
            temp_epsilon_ma,
            self.delta,
            self.alpha,
        )
        
        # need a delta_prime, but we can just default it to 1e-6
        temp_lap_epsilon, temp_lap_delta = privacy_accounting.homogeneous_strong_composition(
                self.eprime,
                0,
                1e-6,
                self.total_queries + 1
            )

        # now we can just combine the two other composed things? i think?
        self.ed_epsilon, self.ed_delta = privacy_accounting.heterogeneous_strong_composition(
                np.asarray([self.gn_epsilon,temp_lap_epsilon], [self.delta,temp_lap_delta],1e-6)
            )
        
        print(temp_epsilon_ma, self.ed_epsilon)
        if self.ed_epsilon > max_epsilon:
            return -1
        return self.aggregate(votes)
