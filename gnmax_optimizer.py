import numpy as np
import aggregate
import get_predicted_labels
import torch
from os.path import isfile
import torch_teachers
import time
import pickle
import globals

start_time = time.time()

SAVEFILE_NAME = "saved/rep_gnmax_points.pkl"

rng = np.random.default_rng()

NUM_POINTS = 20  # changed for our custom checking

_gauss_scale = np.linspace(start=10, stop=200, num=NUM_POINTS)

points = np.asarray([
    _gauss_scale,
])

points = points.transpose()  # get transposed idiot

# initialize file so we don't need to worry about existence later
# (I'm against existential crises :P)
if not isfile(SAVEFILE_NAME):
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(dict(), f)

ds = globals.dataset

loader = torch.utils.data.DataLoader(ds.student_data, shuffle=False, batch_size=256)

if not isfile(f"./saved/{ds.name}_{ds.num_teachers}_teacher_predictions.npy"):
    get_predicted_labels.calculate_prediction_matrix(loader, globals.dataset)

votes = np.load(f"./saved/{ds.name}_{ds.num_teachers}_teacher_predictions.npy", allow_pickle=True)

for point in points:
    gnmax_scale = point

    agg = aggregate.NoisyVectorAggregator(
        gnmax_scale,
        ds,
        noise_fn=rng.normal,
        alpha_set=list(range(2, 21)),
    )
    max_epsilon = 10
   
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

    student_train, student_valid = ds.student_overwrite_labels(labels)

    n, val_acc = torch_teachers.train(student_train, student_valid, ds, epochs=100, batch_size=16, model="student")

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
    # results format is dict of (alpha, p, tau, sigma1, sigma2) : (labeled, label_acc, val_acc, test_acc)
    with open(SAVEFILE_NAME, "rb") as f:
        past_results = pickle.load(f)
    past_results[tuple(point)] = (labeled_len, correct/labeled_len, val_acc, test_acc)
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(past_results, f)

total_time = time.time() - start_time

# print timing info, for long-running processes
print(f"Ran for {NUM_POINTS} rounds;")
print(f"took {total_time // 3600} hours, {total_time // 60 % 60} minutes, and {total_time % 60} seconds;")
print(f"and ended at {time.asctime()}.")
print("Whew!")
