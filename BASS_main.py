"""

Main code for Bayesian Active and Semi-Supervised PATE

based on https://github.com/fbickfordsmith/epig/tree/b11124d2dd48381a5756e14d920d401f1fd3120d/main.py

"""


# Lots of Imports, but I will add as needed so I can better keep track of them all
import torch
from torch.distributions import Gumbel


from datetime import timedelta
import numpy as np
import random
import logging
import globals
from time import time
from BASS_utils import Dictionary
from BASS_model import PyTorchClassificationLaplaceTrainer, FullyConnectedNet, PyTorchClassificationMCDropoutTrainer, MCDropoutFullyConnectedNet

# figure out laplace STILL
import laplace
from laplace import ParametricLaplace
from BASS_utils import *
from BASS_model import *

from privacy_accounting import gnmax_epsilon
from get_predicted_labels import label_by_indices
from aggregate import NoisyMaxAggregator
# Some background functions necessary for the success of this process

# Their code has `get_pytorch_trainer` and `acquire_using_uncertainty` which are probably
# functions that can have homes in other places, but for now, they will be here for ease
# since this branch of the repo is HORRIFYINGLY messy and has a million different things
# going on.

def format_time(seconds: float) -> str:
    time = timedelta(seconds=seconds)
    assert time.days == 0
    hours, minutes, seconds = str(time).split(":")
    return f"{int(hours):02}:{minutes}:{float(seconds):02.0f}"

def get_formatters() -> dict:
    formatters = {
        "step": "{:05}".format,
        "time": format_time,
        "n_labels": "{:04}".format,
        "train_acc": "{:.4f}".format,
        "train_kl": "{:03.4f}".format,
        "train_mae": "{:.4f}".format,
        "train_mse": "{:.4f}".format,
        "train_nll": "{:.4f}".format,
        "val_acc": "{:.4f}".format,
        "val_mae": "{:.4f}".format,
        "val_mse": "{:.4f}".format,
        "val_nll": "{:.4f}".format,
        "test_acc": "{:.4f}".format,
        "test_mae": "{:.4f}".format,
        "test_mse": "{:.4f}".format,
        "test_nll": "{:.4f}".format,
    }
    return formatters

def EPIG_acquire(
        data,
        trainer,
        num_acquisitions,
        input_targs
):
    # might only work with a certain version of torch
    # the documentation specifically calls this "new"
    # unsure if that will throw an error
    # if it does, this is almost analogy to .no_grad()
    # with the addition of disabling `forward-mode AD`
    with torch.inference_mode():
        scores = trainer.estimate_uncertainty(loader=torch.utils.data.DataLoader(data, shuffle=False, batch_size=64),method="epig",seed=random.randint(0,1e6),inputs_targ=input_targs)

    # Use stochastic batch acquisition (https://arxiv.org/abs/2106.12059). <- original comment
    scores = torch.log(scores) + Gumbel(loc=0, scale=1).sample(scores.shape)
    acquired_pool_inds = torch.argsort(scores)[-num_acquisitions :]
    acquired_pool_inds = acquired_pool_inds.tolist()

    return acquired_pool_inds

def acquire_balanced_init(indices,data_object,n_per_label):
    n_labels = data_object.num_labels
    labels_counts = [0] * n_labels
    removed_inds  = []
    
    i = 0
    while len(removed_inds) < n_labels * n_per_label:
        possible = data_object.student_data.dataset.targets[indices[i]]
        if labels_counts[possible] < n_per_label:
            labels_counts[possible] += 1
            removed_inds.append(indices[i])
        i += 1

    return np.asarray(removed_inds)


def main():
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='../saved/logs/BASS.log', level=logging.INFO)
    # setup stuff:
    formatters = get_formatters()
    dat_obj = globals.dataset
    dataset = dat_obj.student_data.dataset
    rng = np.random.default_rng(random.randint(0, int(1e6)))

    agg = NoisyMaxAggregator(50,dat_obj,noise_fn=np.random.normal)
    votes = np.load(f"{globals.SAVE_DIR}/mnist_256_teacher_predictions.npy", allow_pickle=True)
    votes = votes.T

    all_qs = []
    logger.info("setting up data")
    # the pool of student training data that we can pull from!
    data_pool = dat_obj.student_data


    # gotta get some input_targets for epig:
    inp_targ_inds = np.random.choice(data_pool.indices,size = 1000)
    inp_targ_dataset = torch.utils.data.Subset(dataset,inp_targ_inds)
    inp_targs, _ = next(iter(torch.utils.data.DataLoader(inp_targ_dataset,shuffle=False,batch_size=64)))
    data_pool.indices = np.setdiff1d(data_pool.indices,inp_targ_inds)

    # choose a random set for validation and training set
    val_inds = np.random.choice(data_pool.indices,size = 850)

    # apply labels, and then store labels and epsilon costs
    targets, qs = label_by_indices(agg,votes,val_inds)
    all_qs.extend(qs)
    dataset.targets[val_inds] = targets
    
    # calculate epsilon:
    best_eps = None
    for item in agg.alpha_set:
        temp_eps = gnmax_epsilon(all_qs,item,agg.scale,delta=1e-6)
        if best_eps is None:
            best_eps = temp_eps
        elif best_eps > temp_eps:
            best_eps = temp_eps
    print(best_eps)
    
    logger.info(f"Initial epsilon cost: {best_eps:.04f}")

    # remove indices from the pool
    data_pool.indices = np.setdiff1d(data_pool.indices,val_inds)

    # get initial raining data!
    train_inds = acquire_balanced_init(val_inds,dat_obj,5)
    val_inds = np.setdiff1d(val_inds,train_inds)
    X_train = torch.utils.data.Subset(dataset, train_inds)

    val_data = torch.utils.data.Subset(dataset,val_inds)

    valid_loader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=64)

    # save test_loader for later!
    test_loader = torch.utils.data.DataLoader(dat_obj.student_test, shuffle=True, batch_size = 64)
    # Active Learning Loop:

    # their code seems to do the full retraining each time, which is unfortunate
    
    is_first_al_step = True
    start_time = time()
    test_log = Dictionary()

    while True: # change probably
        n_train_labels = len(X_train)
        n_labels_str = f"{n_train_labels:04}_labels"
        is_last_al_step = n_train_labels >= 60 # big number at first?


        logger.info(f"Number of labels: {n_train_labels}")
        logger.info("Setting up trainer")
        # get model:
        net = MCDropoutFullyConnectedNet(input_shape=dataset.data.shape[1:], output_size=dat_obj.num_labels,dropout_rate=0.1,hidden_sizes=[128,128,128]).to(globals.device)

        # get trainer:
        torch_rng = torch.Generator(globals.device).manual_seed(rng.choice(int(1e6)))
        """
        trainer = PyTorchClassificationLaplaceTrainer(
            model = net,
            torch_rng = torch_rng,
            laplace_approx=laplace.Laplace(
                model=net,
                likelihood = 'classification',
                hessian_structure = 'diag',
                subset_of_weights = "all",
                prior_precision = 1
            ),
            likelihood_temperature= "inverse_param_count",
            optimizer = torch.optim.SGD,
            n_optim_steps_min = 0,
            n_optim_steps_max = 200, # VERY different than their value
            n_samples_train = 1,
            n_samples_test = 100,
            n_validations = 40, # every 5 will save validation accuracy
            early_stopping_metric = "val_nll",
            early_stopping_patience = 5000,
            restore_best_model = True
        )
        """
        trainer = PyTorchClassificationMCDropoutTrainer(
            model = net,
            torch_rng = torch_rng,
            optimizer = torch.optim.SGD,
            n_optim_steps_min = 0,
            n_optim_steps_max = 100000,
            n_samples_train = 1,
            n_samples_test = 100,
            n_validations = 1000,
            early_stopping_metric = "val_nll",
            early_stopping_patience = 5000,
            restore_best_model = True
        )

        # train it?
        train_step, train_log = trainer.train(
            train_loader=torch.utils.data.DataLoader(X_train,shuffle=False,batch_size=64), val_loader=valid_loader
        )

        if train_step is not None:
            if train_step < trainer.n_optim_steps_max - 1: # doing 200 epochs?
                logger.info(f"Training stopped early at step {train_step}")
            else:
                logger.warning(f"Training stopped before convergence at step {train_step}")

        if train_log is not None:
            train_log.save_to_csv(globals.SAVE_DIR + "training" + f"{n_labels_str}.csv", formatters)

        np.savetxt(
            globals.SAVE_DIR + "data_indices" + "train.txt", X_train.indices, fmt="%d"
        )


        is_in_save_steps = n_train_labels % 50 == 0 # Hopefully we will be acquiring a lot less data!

        if is_first_al_step or is_last_al_step or is_in_save_steps:
            logger.info("Saving model checkpoint")

            if isinstance(trainer.model, ParametricLaplace):
                model_state = trainer.model.model.state_dict()
            else:
                model_state = trainer.model.state_dict()

            torch.save(model_state, globals.SAVE_DIR + "models" + f"{n_labels_str}.pth")


        logger.info("Testing")
        with torch.inference_mode():
            test_metrics = trainer.test(test_loader)

        test_metrics_str = ", ".join(
            f"{key} = {formatters[f'test_{key}'](value)}" for key, value in test_metrics.items()
        )


        logger.info(f"Test metrics: {test_metrics_str}")

        test_log.append({"n_labels": n_train_labels, **prepend_to_keys(test_metrics, "test")})
        test_log.save_to_csv(globals.SAVE_DIR + "BASS_testing.csv", formatters)

        if is_last_al_step:
            logger.info("Stopping active learning")
            break

        logger.info(
            f"Acquiring 10 label(s) using epig"
        )

        acquired_pool_inds = EPIG_acquire(data_pool, trainer, 10,inp_targs)

        targets, qs = label_by_indices(agg,votes,acquired_pool_inds)
        all_qs.extend(qs)
        dataset.targets[acquired_pool_inds] = targets
        data_pool.indices = np.setdiff1d(data_pool.indices,acquired_pool_inds)
        X_train.indices = np.concat((X_train.indices,acquired_pool_inds))

        #logger.info(f"Epsilon: {gnmax_epsilon(all_qs,agg.alpha,agg.scale,delta=1e-6):.04f}")

        is_first_al_step = False
        
    best_eps = None
    for item in agg.alpha_set:
        temp_eps = gnmax_epsilon(all_qs,item,agg.scale,delta=1e-6)
        if best_eps is None:
            best_eps = temp_eps
        elif best_eps > temp_eps:
            best_eps = temp_eps
    logger.info(f"Final epsilon cost: {best_eps:.04f}")
    run_time = timedelta(seconds=(time() - start_time))
    np.savetxt("../saved/logs/BASS_run_time.txt", [str(run_time)], fmt="%s")



if __name__ == "__main__":
    main()