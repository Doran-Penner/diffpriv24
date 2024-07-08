import numpy as np
import aggregate
import get_predicted_labels
import student
import torch
from os.path import isfile
import torch_teachers
import pickle


rng = np.random.default_rng()

# vars are:
# alpha in (2..10),
# sigma1 in [1, 100],
# sigma2 in [1, 100],
# p in (0,1],
# tau in [1, 100]

NUM_POINTS = 64
SAVEFILE_NAME = "saved/rep_gnmax_points.pkl"

points = np.asarray([
    # change these range values to shrink scope (for optimization)
    rng.choice(np.arange(2,11), size=(NUM_POINTS,)),  # alpha
    rng.uniform(low=0.01, size=(NUM_POINTS,)),  # p
    rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,)),  # tau
    rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,)),  # sigma1
    rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,))  # sigma2
])

points = points.transpose()  # get transposed idiot

all_results = dict()

# initialize file so we don't need to worry about existence later
# (I'm against existential crises :P)
if not isfile(SAVEFILE_NAME):
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(all_results, f)

dataset = 'svhn'
num_teachers = 250

train, valid, _test = get_predicted_labels.load_dataset(dataset, 'student', False)
train = torch.utils.data.ConcatDataset([train, valid])
loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=256)

if not isfile(f"./saved/{dataset}_{num_teachers}_teacher_predictions.npy"):
    get_predicted_labels.calculate_prediction_matrix(loader, get_predicted_labels.device, dataset, num_teachers)


for point in points:
    alpha, p, tau, sigma1, sigma2 = point
    # save:
    # PARAMS,
    # number of labels made,
    # accuracy of those labels,
    # final training accuracy,
    # final validation accuracy,
    # number of epochs

    # ... do stuff
    agg = aggregate.RepeatGNMax(sigma1, sigma2, p, tau, delta=1e-6)
   
    labels = get_predicted_labels.load_predicted_labels(agg, dataset, num_teachers)
    print("FINAL tau usages:", agg.tau_tally)

    correct = 0
    num_datapoints = len(labels)
    unlabeled = 0
    for i, label in enumerate(labels):
        if label == train[i][1]:
            correct += 1
        if label == -1:
            unlabeled += 1
    labeled = num_datapoints-unlabeled
    label_acc = 0
    if labeled != 0:
       label_acc = correct/labeled

    train_set, valid_set, test_set = student.load_and_part_sets(dataset, num_teachers)

    n, val_acc = torch_teachers.train(train_set, valid_set, dataset, device=student.device, epochs=200, model="student")

    # calculate, save results
    # results format is dict of (alpha, p, tau, sigma1, sigma2) : (labeled, label_acc, val_acc)
    all_results[tuple(point)] = (labeled, label_acc, val_acc)

    # and add them to file-saved results for persistent storage
    with open(SAVEFILE_NAME, "rb") as f:
        past_results = pickle.load(f)
        past_results[tuple(point)] = (labeled, label_acc, val_acc)
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(past_results, f)

print("ALL RESULTS THIS ROUND:", all_results)
