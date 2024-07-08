import numpy as np
import aggregate

rng = np.random.default_rng()

# vars are:
# alpha in (2..10),
# sigma1 in [1, 100],
# sigma2 in [1, 100],
# p in (0,1],
# tau in [1, 100]

NUM_POINTS = 3  # TODO change to 64

points = np.asarray([
    rng.choice(np.arange(2,11), size=(NUM_POINTS,)),  # alpha
    rng.uniform(low=0.01, size=(NUM_POINTS,)),  # p
    rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,)),  # tau
    rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,)),  # sigma1
    rng.uniform(low=1e-1, high=256.0, size=(NUM_POINTS,))  # sigma2
])

points = points.transpose()  # get transposed idiot

all_results = []

for point in points:  # TODO
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

    # calculate, save results
    all_results.append((points, None))  # replace None with results

print(all_results)

breakpoint()
