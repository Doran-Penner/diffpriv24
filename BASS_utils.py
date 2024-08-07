"""

This is a catch-all for util functions used by the bits of code we are takin from
https://github.com/fbickfordsmith/epig/tree/b11124d2dd48381a5756e14d920d401f1fd3120d

"""

from typing import Any, Tuple, Union, Sequence, Callable

import torch
from torch import Tensor
from torch.nn.functional import nll_loss
from torch.utils.data import DataLoader

from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path
import globals

# taken from ./src/metrics.py

def accuracy_from_marginals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float], [1,]
    """
    return count_correct_from_marginals(predictions, labels) / len(predictions)  # [1,]


def count_correct_from_marginals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[int], [1,]
    """
    is_correct = torch.argmax(predictions, dim=-1) == torch.argmax(labels,dim=1).to(globals.device)  # [N,]
    return torch.sum(is_correct)  #  [1,]

def nll_loss_from_probs(probs: Tensor, labels: Tensor, **kwargs: Any) -> Tensor:
    """
    Arguments:
        probs: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float]
    """
    probs = torch.clamp(probs, min=torch.finfo(probs.dtype).eps)  # [N, Cl]
    return nll_loss(torch.log(probs), labels, **kwargs)

# ./src/data/utils.py
def get_next(dataloader: DataLoader) -> Union[Tensor, Tuple]:
    try:
        return next(dataloader)
    except:
        dataloader = iter(dataloader)
        return next(dataloader)
    

# taken from ./src/logging.py

class Dictionary(dict):
    def append(self, dictionary: dict) -> None:
        for key in dictionary:
            if key in self:
                self[key] += [dictionary[key]]
            else:
                self[key] = [dictionary[key]]

    def extend(self, dictionary: dict) -> None:
        for key in dictionary:
            if key in self:
                self[key] += dictionary[key]
            else:
                self[key] = dictionary[key]

    def concatenate(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            if isinstance(dictionary[key][0], np.ndarray):
                dictionary[key] = np.concatenate(dictionary[key])
            elif isinstance(dictionary[key][0], Tensor):
                if dictionary[key][0].ndim == 0:
                    dictionary[key] = torch.tensor(dictionary[key])
                else:
                    dictionary[key] = torch.cat(dictionary[key])
            else:
                raise TypeError

        return dictionary

    def numpy(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = dictionary[key].numpy()

        return dictionary

    def torch(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = torch.tensor(dictionary[key])

        return dictionary

    def subset(self, inds: Sequence) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = dictionary[key][inds]

        return dictionary

    def save_to_csv(self, filepath: Path, formatting: Union[Callable, dict] = None) -> None:
        table = pd.DataFrame(self)

        if callable(formatting):
            table = table.applymap(formatting)

        elif isinstance(formatting, dict):
            for key in formatting:
                if key in self:
                    table[key] = table[key].apply(formatting[key])

        table.to_csv(filepath, index=False)

    def save_to_npz(self, filepath: Path) -> None:
        np.savez(filepath, **self)


def prepend_to_keys(dictionary: dict, string: str) -> dict:
    return {f"{string}_{key}": value for key, value in dictionary.items()}