import torch
from datasets import make_dataset

# below equivalent to "pragma once" so we don't re-calculate dataset every time
if "been_run" not in vars():
    been_run = True
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print("using device", device)

    # note: here's our single place to hard-code the dataset & num_teachers,
    # if/when we change it we should be able to just change this
    # (check that to be sure though!)
    dataset = make_dataset("svhn", 250, seed=96)
