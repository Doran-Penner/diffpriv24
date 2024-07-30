import torch
import numpy as np
from torch_teachers import train
import globals
import random
import aggregate
from get_predicted_labels import label_by_indices
from privacy_accounting import gnmax_epsilon
from models import BayesCNN
from bayesian_model import BayesianNet, BBB3Conv3FC
from torch.optim import Adam, lr_scheduler
from acquisition_funcs import BatchBALD
from matplotlib import pyplot as plt
import Bayes_utils as utils
from bayesian_learning import train_model, validate_model

def calculate_test_accuracy(network, test_data):
    """
    Function to calculate the accuracy of the student model on the test data
    :param network: student model
    :param test_data: dataset containing the test data
    :returns: number representing the accuracy of the student model on the test data
    """
    network.eval()
    batch_size = 64
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
    accs = []
    for batch_xs, batch_ys in test_loader:
        batch_xs = batch_xs.to(globals.device,dtype=torch.float32)
        batch_ys = batch_ys.to(globals.device,dtype=torch.float32)
        preds = network(batch_xs)
        # preds is a tuple of (tensor[64,10],) for some reason
        accs.append((torch.argmax(preds[0],dim=1) == torch.argmax(batch_ys,dim=1)).float())
    acc = torch.cat(accs).mean()
    return acc  # we don't see that :)



def student_train(training_data,valid_data, lr_start=1e-3,epochs=70,batch_size=16,net=BBB3Conv3FC):
    """
    based on:
    https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/main_bayesian.py
    """

    train_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=batch_size)
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    model = net(globals.dataset).to(globals.device) # same as torch_teachers.train (but w/o the dat_obj passed)

    criterion = utils.ELBO(len(training_data)).to(globals.device)
    optimizer = Adam(model.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_max = np.inf
    for e in range(epochs):

        train_loss, train_acc, train_kl = train_model(model, optimizer, criterion, train_loader, epoch=e, num_epochs=epochs)
        valid_loss, valid_acc = validate_model(model, criterion, valid_loader, epoch=e, num_epochs=epochs)
        lr_sched.step(valid_loss)


        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            e, train_loss, train_acc, valid_loss, valid_acc, train_kl))
        
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            torch.save(model.state_dict(),f"{globals.SAVE_DIR}/{globals.dataset.name}_bayesian_student.ckp")
            valid_loss_max = valid_loss

    st_dict = torch.load(f"{globals.SAVE_DIR}/{globals.dataset.name}_bayesian_student.ckp",map_location=globals.device)
    model.load_state_dict(st_dict)

    return model, valid_loss_max


def active_learning(network=BBB3Conv3FC,acquisition_iterations=10,initial_size=100,acquisition_method=BatchBALD,num_acquisitions=10,print_summary=True,epochs=10):

    """
    Function to do active learning with a BayesianNet object. (a, hopefully, cleaner version of active_train)
    


    :param network: The class of student you wish to train
    :param acquisition_iterations: the number of rounds of acquisitions you wish to do
    :param initial_size: the size of the initial training set before you choose new points
    :param acquisition_method: the class of acquirer you wish to use to pick points to label
    :param num_acquisitions: the number of acquisitions per acquisition iteration
    :param print_summary: a boolean to determine if we print out the summary at the end

    :returns model: a fully trained student model
    :returns test_dict: a dictionary of form {"epsilon":[],"test_acc":[]} that stores values per acquisition iteration
    """

    # start with the initial training!
    dat_obj = globals.dataset
    dataset = dat_obj.student_data.dataset

    data_dep_eps_costs = []

    votes = np.load(f"{globals.SAVE_DIR}/{dat_obj.name}_{dat_obj.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    votes = votes.T

    agg = aggregate.NoisyMaxAggregator(40,dat_obj,noise_fn=np.random.normal) # TODO possibly change this for abstraction later

    # for testing purposes, to be removed later
    test_dict = {"epsilon":[],"test_acc":[],"valid_loss":[]}

    # the pool of student training data that we can pull from!
    data_pool = dat_obj.student_data

    # initial training set randomly chosen
    sample_indices = np.random.choice(data_pool.indices,size = initial_size)
    X_train = torch.utils.data.Subset(dataset, sample_indices)

    # remove the initial training data from data_pool
    data_pool.indices = np.setdiff1d(data_pool.indices,sample_indices)

    # get labels for the training data and then store renyi epsilon cost
    Y_train, train_qs = label_by_indices(agg,votes,X_train.indices)
    X_train.dataset.labels[X_train.indices] = Y_train
    data_dep_eps_costs += train_qs

    
    # get the validation set!
    val_inds = np.random.choice(data_pool.indices,size = 2000) # smallish validation set :)
    val_data = torch.utils.data.Subset(dataset,val_inds)

    # remove the valid data from the data_pool
    data_pool.indices = np.setdiff1d(data_pool.indices,val_inds)

    # label the validation data and store the labels/epsilon costs
    val_ys, val_qs = label_by_indices(agg,votes,val_data.indices)
    data_dep_eps_costs += val_qs
    val_data.dataset.labels[val_data.indices] = val_ys

    model, valid_loss = student_train(X_train,val_data,epochs=epochs,net=network)

    # saving relevant information for later (mainly for testing purposes)
    test_dict["valid_loss"].append(valid_loss)
    test_dict["epsilon"].append(gnmax_epsilon(data_dep_eps_costs,alpha=agg.alpha,sigma=agg.scale,delta=1e6))
    test_dict["test_acc"].append(calculate_test_accuracy(model,dat_obj.student_test))
    
    # size of the subset of the pool we wish to acquire from
    pool_subset_size = 1000
    
    # this stores the algorithm we wish to use for the acquisition of datapoints
    acquirer = acquisition_method(num_acquisitions,dat_obj,subset_size=pool_subset_size)

    for round in range(acquisition_iterations):
        print("Round: ", round)

        # get the best indices using our acquisition function
        # np array
        selected_indices = acquirer.select_batch(model,data_pool)

        # add these indices to X_train, and remove them from data_pool
        # make sure that having out-of-order indices doesn't severely mess things up
        X_train.indices = np.concat((X_train.indices,selected_indices))
        data_pool.indices = np.setdiff1d(data_pool.indices,selected_indices)

        new_labels, new_qs = label_by_indices(agg,votes,selected_indices)

        X_train.dataset.labels[selected_indices] = new_labels
        data_dep_eps_costs += new_qs

        # retrain the student!
        model, valid_loss = student_train(X_train,val_data, epochs=epochs,net=network)

        test_dict["valid_loss"].append(valid_loss)
        test_dict["test_acc"].append(calculate_test_accuracy(model,dat_obj.student_test))
        test_dict["epsilon"].append(gnmax_epsilon(data_dep_eps_costs,alpha=agg.alpha,sigma=agg.scale,delta=1e6)) # TODO possibly change this for abstraction later!

    if print_summary:
        print_assessment(test_dict,initial_size,acquisition_iterations,num_acquisitions)
    

    return model, test_dict


def print_assessment(test_dict,initial_size,acquisition_iterations,num_acquisitions):
    acqusitions = np.linspace(initial_size,initial_size+acquisition_iterations * num_acquisitions, num = num_acquisitions)
    accuracies = test_dict["test_acc"]
    epsilons = test_dict["epsilon"]
    # print out results:
    print("acquisitions:\ttest_acc\tepsilon")
    for i in range(len(test_dict["epsilon"])):
        print(f"{acqusitions[i]}\t\t{accuracies[i]}\t\t{epsilons[i]}")

    # plot it because plots are fun :)
    fig, ax1 = plt.subplots()

    color = 'tab:purple'
    ax1.set_xlabel("Acquisition Iterations")
    ax1.set_ylabel("Test Accuracy", color=color)
    ax1.plot(acqusitions,test_dict["test_acc"],color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:cyan'
    color = 'tab:blue'
    ax2.set_ylabel('epsilon_cost', color=color)
    ax2.plot(acqusitions, test_dict["epsilon"], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout() 
    plt.show()

    

def active_train(network=BayesCNN,dropout_iterations=100,acquisitions=25,acquisitions_iterations=36, initial_size=100):
    """
    Function to do active learning with the BayesCNN based on the same code as the 
    BayesCNN architecture (not to be used with BayesianNet)

    :param network: A class of neural networks to train
    :param dat_obj: Dataset to learn from
    :param dropout_iterations: number of iterations for MC dropout
    :param acqusitions: number of data points to acquire in each round
    :param acquisition_iterations: number of rounds
    :param initial_size: the size of the initial training data
    """
    
    # with the default parameters we will have 1000 labeled data points, which is in-line with
    # the paper on image data learning, but we will probably want/need to do more

    # first, we need to break up the student data into pool and initial data
    # the original code attempts to get balanced data, which might not be possible in terms of
    # differential privacy, so we will just take a larger initial sample to deal with that issue

    # full dataset to be used later
    dataset = globals.dataset
    data = dataset.student_data.dataset

    full_qs = []

    # teacher votes (assuming the votes have already been given just not aggregated)
    votes = np.load(f"{globals.SAVE_DIR}/{dataset.name}_{dataset.num_teachers}_teacher_predictions.npy", allow_pickle=True)
    votes = votes.T
    # abstraction later maybe
    agg = aggregate.NoisyMaxAggregator(40,dataset,noise_fn=np.random.normal)

    # for the plot later
    val_accs = []
    # not sure if i want/need test_acc, but for purposes It will be left
    test_accs = []
    # also not super helpful later on, but this will keep track of epsilon as we go
    eps = []


    data_pool,valid_data = torch.utils.data.random_split(dataset.student_data, [0.9, 0.1])

    X_train = torch.utils.data.Subset(data,data_pool.indices[:initial_size]) # take the indices of the first initial_size data points
    data_pool.indices = data_pool.indices[initial_size:]
    valid_data = torch.utils.data.Subset(data,valid_data.indices)

    # labels for the valid and training sets to start off with
    valid_labels, v_qs = label_by_indices(agg,votes,valid_data.indices)
    Y_train, train_qs = label_by_indices(agg,votes,X_train.indices)

    full_qs.extend(v_qs)
    full_qs.extend(train_qs)

    # put the found labels onto the dataset
    X_train.dataset.labels[X_train.indices] = Y_train
    valid_data.dataset.labels[valid_data.indices] = valid_labels

    # initial training before active loop:
    
    model, val_acc = train(X_train, valid_data, dataset, epochs=200, model="student",net=network)


    val_accs.append(val_acc)

    # again, not sure how much we want to use this, but i'll leave it here for now!
    # also, they have it in the paper, so I want to keep it for consistency, at least
    # for the time being, just so we can maybe get a pretty graph!
    test_accs.append(calculate_test_accuracy(model,dataset.student_test))

    # now to start active learning loop!
    for round in range(acquisitions_iterations):

        print("Acquisition Round: ", round)
        # Note, we don't want to calculate the acquistion function for every single point
        # since that would be pretty computationally expensive (maybe try it some time)
        # so we will take a subset of the pool data to check

        pool_subset_size = 2000 # same as the code for the paper but might want to be tweaked
        pool_subset = torch.tensor(random.sample(data_pool.indices,pool_subset_size))

        # subset object of the pool subset to calculate acquisition function on.
        pool_dropout = torch.utils.data.Subset(data,pool_subset)
        # this will store the scores and entropy for the dropout iterations
        all_scores = torch.zeros(size=(pool_subset_size,dataset.num_labels),dtype=torch.float32)
        all_entropy = torch.zeros(size=(pool_subset_size,),dtype=torch.float32)

        for d in range(dropout_iterations):

            print("\tDropout iteration: ", d)
            dropout_scores = predict_stochastic(model,pool_dropout)
            dropout_scores = dropout_scores.to(torch.device('cpu'),dtype=torch.float32)

            # get the scores added up for later calculations
            all_scores = torch.add(all_scores,dropout_scores)

            dropout_score_log = torch.log2(dropout_scores)
            entropy_compute = - torch.mul(dropout_scores,dropout_score_log)
            entropy_per_dropout = torch.sum(entropy_compute,axis=1)
            all_entropy = torch.add(all_entropy, entropy_per_dropout)

        avg_scores = torch.div(all_scores,dropout_iterations)
        avg_scores_log = torch.log2(avg_scores)
        entropy_avg_scores = - torch.mul(avg_scores,avg_scores_log)
        entropy_avg_scores = torch.sum(entropy_avg_scores,axis=1)

        average_entropy = torch.div(all_entropy, dropout_iterations)

        all_bald = torch.sub(entropy_avg_scores, average_entropy)

        all_bald = all_bald.flatten()

        
        acquired_indices = all_bald.argsort()[-acquisitions:]
        # since we are acquiring the indices of the list of indices we need to undo
        # that process so we have the original dataset indices to label from
        dataset_indices = pool_dropout.indices[acquired_indices]

        new_labels, new_qs = label_by_indices(agg,votes,dataset_indices)

        X_train.indices += dataset_indices

        X_train.dataset.labels[dataset_indices] = new_labels
        full_qs.extend(new_qs)

        eps.append(gnmax_epsilon(full_qs,agg.alpha,agg.scale,1e-6))

        # delete the chosen items from the data_pool
        data_pool.indices = [x for i,x in enumerate(data_pool.indices) if i not in dataset_indices]
        
        # retrain the model!
        model, val_acc = train(X_train, valid_data, dataset, epochs=200, model="student",net=network)
        
        val_accs.append(val_acc)

        # again, not sure how much we want to use this, but i'll leave it here for now!
        test_accs.append(calculate_test_accuracy(model,dataset.student_test))
    
    print_assessment_active_train(eps,val_accs,test_accs)

    # finally, we just have to sum-up the epsilon (for now it will be un-noised)
    # and return the model!
    epsilon = gnmax_epsilon(full_qs,agg.alpha,agg.scale,1e-6)


    return model, epsilon

def print_assessment_active_train(eps,valid_accuracies,test_accuracies):
    # a pretty print function for active_train

    print("acquisition:\tvalidation\ttest\t\tepsilon")
    for i in range(len(valid_accuracies)):
        print(f"{i}\t\t{valid_accuracies[i]}\t\t{test_accuracies[i]}\t\t{eps[i]}")

def predict_stochastic(model,X):
    """
    Function to output the dropout scores of a set of data
    This can be made more efficient, but for the time being
    I am going to just keep it this way.
    """
    X_data = torch.utils.data.DataLoader(X, shuffle=True,batch_size=64)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_xs, _ in X_data:
            batch_xs = batch_xs.to(globals.device,dtype=torch.float32)
            preds = model(batch_xs)
            pred_list.append(preds.to(torch.device('cpu')))

    ret_tensor = torch.cat(pred_list,dim=0)
    
    return ret_tensor

def old_student_train(training_data,lr=1e-3, epochs=70,batch_size=16,net=BayesianNet):
    """
    function to train a student of a certain class. For the time being this function does not deal
    with validation data, and just takes the final epoch. This is in line with the code found at
    https://github.com/french-paragon/BayesianMnist/blob/master/viExperiment.py

    Not used anymore (but keeping it around in case we want to use it again later)
    """

    train_loader = torch.utils.data.DataLoader(training_data, shuffle=True, batch_size=batch_size)

    model = net(globals.dataset).to(globals.device) # same as torch_teachers.train (but w/o the dat_obj passed)

    N = len(training_data) # number of data points we are training on!

    loss = torch.nn.NLLLoss(reduction='mean') #negative log likelihood will be part of the ELBO
    optimizer = Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    for n in np.arange(epochs) :
        if n % 5 ==4:
            print("Epoch:", n+1)

        for i, batch in enumerate(train_loader) :
            batch_xs, batch_ys = batch

            pred = model(batch_xs, stochastic=True)

            logprob = loss(pred, batch_ys)
            l = N*logprob

            modelloss = model.evalAllLosses()
            l += modelloss
            
            optimizer.zero_grad()
            l.backward()
            
            optimizer.step()

    return model



#def main():
#    # this is where we set the parameters that are used by the functions in this file (ie, if we
#    # want to use a different database, we would change it here)
#    ds = globals.dataset
#    dataset_name = ds.name
#    num_teachers = ds.num_teachers
#
#    labels = np.load(f"{globals.SAVE_DIR}/{dataset_name}_{num_teachers}_agg_teacher_predictions.npy", allow_pickle=True)
#
#    train_set, valid_set = ds.student_overwrite_labels(labels)
#    test_set = ds.student_test
#
#    n, val_acc = train(train_set, valid_set, ds, epochs=200, model="student")
#
#    print(f"Validation Accuracy: {val_acc:0.3f}")
#    test_acc = calculate_test_accuracy(n, test_set)
#    print(f"Test Accuracy: {test_acc:0.3f}")
#    torch.save(n.state_dict(), f"{globals.SAVE_DIR}/{dataset_name}_student_final.ckp")

if __name__ == '__main__':
    active_learning()
