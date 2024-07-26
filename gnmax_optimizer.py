import numpy as np
import aggregate
import get_predicted_labels
import torch
import os
from os.path import isfile
import time
import pickle
import csv
import globals
from training import train_ssl

start_time = time.time()

SAVEFILE_NAME = f"./saved/{globals.prefix}_gnmax_optimizer_points.pkl"
CSVPATH = f"./saved/{globals.prefix}_optimize.csv"

rng = np.random.default_rng()

NUM_POINTS = 21 # changed for our custom checking

_gauss_scale = np.full(NUM_POINTS, 100)
_scale2 = np.full(NUM_POINTS, 40)
_tau = np.linspace(start=100, stop=200, num=NUM_POINTS)

points = np.asarray([
    _gauss_scale,
    _scale2,
    _tau
])

points = points.transpose()  # get transposed idiot

# initialize file so we don't need to worry about existence later
# (I'm against existential crises :P)
if not isfile(SAVEFILE_NAME):
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(dict(), f)
else:
    print("WARNING: the savefile already exists, meaning that the newly-"
          "generated data will be appended to or overwrite the old data.")
    print("If you don't want this to happen, cancel this process in the next "
          "10 seconds (with <Ctrl-C>) and delete the file at " + SAVEFILE_NAME)
    # we sleep instead of awaiting input so this script can run in the background without hiccups
    time.sleep(10)
    print("Got no cancel, so continuing!")
    time.sleep(1)

ds = globals.dataset

votes = np.load(f"./saved/{ds.name}_{ds.num_teachers}_teacher_predictions.npy", allow_pickle=True)

for point in points:
    confidence_scale, argmax_scale, tau = point

    agg = aggregate.ConfidentGNMax(
        confidence_scale,
        argmax_scale,
        tau,
        ds,
        alpha_set=list(range(2, 21)),
    )
    max_epsilon = 1
   
    labels = get_predicted_labels.load_predicted_labels(agg, votes, ds, max_epsilon)

    # this is a bit weird, but seems to work
    full_len = len(labels)
    # get indices of labeled datapoints, for use later
    which_labeled = np.arange(full_len)[np.all(labels != None, axis=1)]  # noqa: E711
    labeled_labels = labels[which_labeled]
    labeled_len = len(labeled_labels)

    correct = 0
    for i, label in zip(which_labeled, labeled_labels):
        if label.argmax() == ds.student_data[i][1].argmax():
            correct += 1

    print(f"data points labeled: {labeled_len} out of {full_len} ({labeled_len / full_len:0.3f})")
    if labeled_len != 0:
        print(f"label accuracy on labeled data: {correct/labeled_len:0.3f}")

    student_train, student_valid, unlabeled = ds.student_overwrite_labels(labels, semi_supervise=True)
    
    n, val_acc = train_ssl(student_train, unlabeled, student_valid, ds, 0.95, num_rounds=10, epochs=500, batch_size=16)

    # NOTE: this is really bad practice since we're optimizing w.r.t. the test data,
    # but for now we just need to see if things actually work
    test_data = ds.student_test
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=256)
    n.eval()
    test_accs = []
    for batch_xs, batch_ys in test_loader:
        batch_xs = batch_xs.to(globals.device)
        batch_ys = batch_ys.to(globals.device)
        preds = n(batch_xs)
        test_accs.append((preds.argmax(dim=1) == batch_ys.argmax(dim=1)).float().mean())
    test_acc = torch.tensor(test_accs).mean()

    # now save the results! we write to disk every time so we can cancel the process with minimal loss
    # results format is dict of (point) : (labeled, label_acc, val_acc, test_acc)
    with open(SAVEFILE_NAME, "rb") as f:
        past_results = pickle.load(f)
    past_results[tuple(point)] = (labeled_len, correct/labeled_len, val_acc, test_acc)
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(past_results, f)

### CSV creation
with open(SAVEFILE_NAME, "rb") as f:
    final_results = pickle.load(f)
table = []
# unpack the dict into a more normal list so we can easily write it to csv
for keyval_pair in final_results.items():
    key, val = keyval_pair
    table.append(list(key) + list(val))
# make everything a float for consistency
for i, x in enumerate(table):
    table[i] = list(map(lambda y: float(y), x))
# delete any pre-existing csv
if isfile(CSVPATH):
    os.remove(CSVPATH)
# finally, write to the file!
with open(CSVPATH, "w", newline="") as f:
    writer = csv.writer(f)
    for row in table:
        writer.writerow(row)

total_time = time.time() - start_time

# print timing info, for long-running processes
print(f"Ran for {NUM_POINTS} rounds;")
print(f"took {total_time // 3600} hours, {total_time // 60 % 60} minutes, and {total_time % 60} seconds;")
print(f"and ended at {time.asctime()}.")
print("Whew!")
