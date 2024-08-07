"""

This is a catch-all for util functions used by the bits of code we are takin from
https://github.com/fbickfordsmith/epig/tree/b11124d2dd48381a5756e14d920d401f1fd3120d

"""

from typing import Any, Tuple, Union, Sequence, Callable

import math

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
    is_correct = torch.argmax(predictions, dim=-1) == labels  # [N,]
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

# taken from ./src/math.py

def logmeanexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """
    Numerically stable implementation of log(mean(exp(x))).
    """
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])


def accuracy_from_conditionals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, K, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float], [1,]
    """
    return count_correct_from_conditionals(predictions, labels) / len(predictions)  # [K,]


def count_correct_from_conditionals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, K, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[int], [1,]
    """
    is_correct = torch.argmax(predictions, dim=-1) == labels[:, None]  # [N, K]
    return torch.sum(is_correct, dim=0)  #  [K,]

# ./src/models/utils.py
def compute_conv_output_size(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    n_output_channels: int,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    width = compute_conv_output_width(input_width, kernel_sizes, strides, padding, dilation)

    return n_output_channels * (width**2)


def compute_conv_output_width(
    input_width: int,
    kernel_sizes: Sequence[int],
    strides: Sequence[int],
    padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    References:
        https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """
    width = input_width

    for kernel_size, stride in zip(kernel_sizes, strides):
        width = width + (2 * padding) - (dilation * (kernel_size - 1)) - 1
        width = math.floor((width / stride) + 1)

    return width