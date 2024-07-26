import torch
import numpy as np
from torch_teachers import train
import globals
import random
import aggregate
from get_predicted_labels import label_by_indices
from privacy_accounting import gnmax_epsilon
from models import BayesCNN



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
        accs.append((preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float())
    acc = torch.cat(accs).mean()
    return acc  # we don't see that :)


def active_train(network=BayesCNN,dropout_iterations=100,acquisitions=25,acquisitions_iterations=36, initial_size=100):
    """
    Function to do active learning with the BayesCNN based on the same code as the 
    BayesCNN architecture

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
    
    

    # finally, we just have to sum-up the epsilon (for now it will be un-noised)
    # and return the model!
    epsilon = gnmax_epsilon(full_qs,agg.alpha,agg.scale,1e-6)


    return model, epsilon

def print_assessment(eps,valid_accuracies,test_accuracies):
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
    active_train()
