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

NUM_POINTS = 15  # changed for our custom checking

_gnmax_scale = np.full((NUM_POINTS,), 50)
_gnmax_eps = np.full((NUM_POINTS,), 5)
_max_num = np.full((NUM_POINTS,), 1000)
_lap_scale = np.linspace(start=10, end=150, num=NUM_POINTS)

points = np.asarray([
    _gnmax_scale,
    _gnmax_eps,
    _max_num,
    _lap_scale,
])

points = points.transpose()  # get transposed idiot

# initialize file so we don't need to worry about existence later
# (I'm against existential crises :P)
if not isfile(SAVEFILE_NAME):
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(dict(), f)

ds = globals.dataset
num_teachers = ds.num_teachers

student_data = globals.dataset.student_data
loader = torch.utils.data.DataLoader(student_data, shuffle=False, batch_size=256)

if not isfile(f"./saved/{ds.name}_{num_teachers}_teacher_predictions.npy"):
    get_predicted_labels.calculate_prediction_matrix(loader, globals.dataset)


for point in points:
    gnmax_scale, gnmax_eps, max_num, lap_scale = point
    # save:
    # PARAMS,
    # number of labels made,
    # accuracy of those labels,
    # final training accuracy,
    # final validation accuracy,
    # number of epochs

    # ... do stuff
    agg = aggregate.PartRepeatGNMax(
        # ignoring p, tau, confident, alpha_set
        GNMax_scale=gnmax_scale,
        p=1,
        tau=50,
        dat_obj=globals.dataset,
        max_num=max_num,
        confident=False,
        lap_scale=lap_scale,
        GNMax_epsilon=gnmax_eps,
        alpha_set=list(range(2,21))
    )
    max_epsilon = 10
   
    labels = get_predicted_labels.load_predicted_labels(agg, ds, max_epsilon)
    print("FINAL tau usages:", agg.tau_tally)

    correct = 0
    num_datapoints = len(labels)
    unlabeled = 0
    for i, label in enumerate(labels):
        if label == student_data[i][1]:
            correct += 1
        if label is not None:
            unlabeled += 1
    labeled = num_datapoints-unlabeled
    label_acc = 0
    if labeled != 0:
       label_acc = correct/labeled

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
        test_accs.append((preds.argmax(dim=1) == batch_ys).float().mean())
    test_acc = torch.tensor(test_accs).mean()

    # now save the results! we write to disk every time so we can cancel the process with minimal loss
    # results format is dict of (alpha, p, tau, sigma1, sigma2) : (labeled, label_acc, val_acc, test_acc)
    with open(SAVEFILE_NAME, "rb") as f:
        past_results = pickle.load(f)
    past_results[tuple(point)] = (labeled, label_acc, val_acc, test_acc)
    with open(SAVEFILE_NAME, "wb") as f:
        pickle.dump(past_results, f)

total_time = time.time() - start_time

# print timing info, for long-running processes
print(f"Ran for {NUM_POINTS} rounds;")
print(f"took {total_time // 3600} hours, {total_time // 60 % 60} minutes, and {total_time % 60} seconds;")
print(f"and ended at {time.asctime()}.")
print("Whew!")
