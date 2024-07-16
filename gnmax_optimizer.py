import numpy as np
import aggregate
import get_predicted_labels
import student
import torch
from os.path import isfile
import torch_teachers
import privacy_accounting
import helper
import time
import pickle

start_time = time.time()

SAVEFILE_NAME = "saved/rep_gnmax_points.pkl"

### Using this info
# as long as you don't interfere with the script as it's reading/writing,
# you can see the results with the following code
# (best to just do this in a REPL):

# with open(SAVEFILE_NAME, "rb") as f:
#     results = pickle.load(f)

# # find the best point by validation accuracy:
# best_point = max(results, key=(lambda x: results.get(x)[3]))
# print("best point:", best_point)
# print("values:", results[best_point])


rng = np.random.default_rng()

# vars are:
# alpha in (2..10),
# sigma1 in [1, 100],
# sigma2 in [1, 100],
# p in (0,1],
# tau in [1, 100]

NUM_POINTS = 40  # changed for our custom checking

# points = np.asarray([
#     # change these range values to shrink scope (for optimization)
#     rng.choice(np.arange(2,11), size=(NUM_POINTS,)),  # alpha
#     rng.uniform(low=0.01, size=(NUM_POINTS,)),  # p
#     rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,)),  # tau
#     rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,)),  # sigma1
#     rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,))  # sigma2
# ])

_alpha = np.full((NUM_POINTS,), 3)
_p = np.full((NUM_POINTS,), 0.75)
_tau = _p * 50
_sigma1 = np.arange(1,41) * 5 * _p  # [5, 10, ..., 200]
_sigma2 = np.full((NUM_POINTS,), 50)

points = np.asarray([
    _alpha,
    _p,
    _tau,
    _sigma1,
    _sigma2
])

points = points.transpose()  # get transposed idiot

# initialize file so we don't need to worry about existence later
# (I'm against existential crises :P)
if not isfile(SAVEFILE_NAME):
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(dict(), f)

dataset = 'svhn'
num_teachers = 250

train, valid, _test = get_predicted_labels.load_dataset(dataset, 'student', False)
train = torch.utils.data.ConcatDataset([train, valid])
loader = torch.utils.data.DataLoader(train, shuffle=False, batch_size=256)

if not isfile(f"./saved/{dataset}_{num_teachers}_teacher_predictions.npy"):
    get_predicted_labels.calculate_prediction_matrix(loader, helper.device, dataset, num_teachers)


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
    agg = aggregate.RepeatGNMax(
        sigma1,
        sigma2,
        p,
        tau,
        delta=1e-6,
        distance_fn=helper.swing_distance,
        epsilon_prime=privacy_accounting.epsilon_prime,
    )
   
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

    n, val_acc = torch_teachers.train(train_set, valid_set, dataset, device=helper.device, epochs=100, batch_size=16, model="student")

    # compute our final accuracy metric on *true labels* of validation data
    _train_data, valid_data, _test_data = helper.load_dataset(dataset, "student")
    valid_loader = torch.utils.data.DataLoader(valid_data, shuffle=True, batch_size=256)
    n.eval()
    true_val_accs = []
    for batch_xs, batch_ys in valid_loader:
        batch_xs = batch_xs.to(helper.device)
        batch_ys = batch_ys.to(helper.device)
        preds = n(batch_xs)
        true_val_accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
    true_val_acc = torch.tensor(true_val_accs).mean()

    # now save the results! we write to disk every time so we can cancel the process with minimal loss
    # results format is dict of (alpha, p, tau, sigma1, sigma2) : (labeled, label_acc, val_acc, true_val_acc)
    with open(SAVEFILE_NAME, "rb") as f:
        past_results = pickle.load(f)
    past_results[tuple(point)] = (labeled, label_acc, val_acc, true_val_acc)
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(past_results, f)

total_time = time.time() - start_time

# print timing info, for long-running processes
print(f"Ran for {NUM_POINTS} rounds;")
print(f"took {total_time // 3600} hours, {total_time // 60 % 60} minutes, and {total_time % 60} seconds;")
print(f"and ended at {time.asctime()}.")
print("Whew!")
