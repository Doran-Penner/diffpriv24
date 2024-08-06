"""

This file is meant to allow us integrating with https://github.com/fbickfordsmith/epig/tree/main
We need to over-write the "ActiveLearningData" class so that we can have teacher data
as well as noisy labeling. 

The code is initially taken from the repo at ./src/data/datasets/active_learning.py

"""

from typing import Callable, Sequence, Union

import numpy as np
from numpy.random import Generator
from omegaconf import ListConfig
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import math

from helper import data_dependent_cost

# make sure we change this to fit the library import!
from src.data.datasets.base import BaseDataset
from src.typing import ConfigDict, ConfigList


class ActiveLearningData:
    def __init__(
        self,
        dataset: BaseDataset,
        device: str,
        rng: Generator,
        batch_sizes: ConfigDict,
        label_counts_main: ConfigDict,
        label_counts_test: Union[ConfigDict, ConfigList, int],
        class_map: Union[ConfigDict, ConfigList] = None,
        loader_kwargs: ConfigDict = None,
    ) -> None:
        self.num_labels = 10
        # NOTE As we are writing, we need to change main to be teacher, and test to be student + test
        # our actual code leaves 10% of the "test" data as proper test data and uses the rest as student data.
        og_train = dataset(train=True) # was self.main_dataset
        og_test = dataset(train=False) # was self.test_datasets
        num_teachers = 256 # for now, this is a static variable

        # From dataset._MNIST()
        all_teach_train, all_teach_valid = torch.utils.data.random_split(
            og_train,
            [0.8, 0.2],
            generator=self._generator,
        )
        train_size = len(all_teach_train)
        valid_size = len(all_teach_valid)

                # then partition them into num_teachers selections
        train_partition = [
            math.floor(train_size / num_teachers) + 1
            for i in range(train_size % num_teachers)
        ] + [
            math.floor(train_size / num_teachers)
            for i in range(num_teachers - (train_size % num_teachers))
        ]
        valid_partition = [
            math.floor(valid_size / num_teachers) + 1
            for i in range(valid_size % num_teachers)
        ] + [
            math.floor(valid_size / num_teachers)
            for i in range(num_teachers - (valid_size % num_teachers))
        ]

        # now assign self variables to those (these are what the "user" will access!)
        self.teach_train = torch.utils.data.random_split(
            all_teach_train, train_partition, generator=self._generator
        )
        self.teach_valid = torch.utils.data.random_split(
            all_teach_valid, valid_partition, generator=self._generator
        )
        self.teach_test = og_test

        student_data_len = math.floor(len(og_test) * 0.9)

        # for our purposes, these two attributes are equivalent
        # to the github repo's self.main_dataset and self.test_dataset
        self.student_data = torch.utils.data.Subset(
            og_test, np.arange(student_data_len)
        )
        self.student_test = torch.utils.data.Subset(
            og_test, np.arange(student_data_len, len(og_test))
        )
        # end from datasets.MNIST_


        self.main_inds = {}

        free_inds = np.arange(len(self.student_data), dtype=int)

        for subset in label_counts_main.keys():
            selected_inds = initialize_indices(
                self.student_data.targets[free_inds], label_counts_main[subset], rng
            )  # Index into free_inds
            selected_inds = free_inds[selected_inds]  # Index into self.main_dataset

            self.main_inds[subset] = selected_inds.tolist()

            free_inds = np.setdiff1d(free_inds, selected_inds)

        self.test_inds = initialize_indices(self.student_test.targets, label_counts_test, rng)
        self.test_inds = self.test_inds.tolist() # why list here?

        if class_map is not None:
            self.student_data = map_classes(self.student_data, class_map)
            self.student_test = map_classes(self.student_test, class_map)

        self.batch_sizes = batch_sizes
        self.device = device
        self.loader_kwargs = loader_kwargs if loader_kwargs is not None else {}

    @property
    def n_train_labels(self) -> int:
        return len(self.main_inds["train"])

    def convert_datasets_to_numpy(self) -> None:
        self.teach_train = self.teach_train.numpy()
        self.teach_test = self.teach_test.numpy()
        self.teach_valid = self.teach_valid.numpy()
        self.student_data = self.student_data.numpy()
        self.student_test = self.student_test.numpy()

    def convert_datasets_to_torch(self) -> None:
        self.teach_train = self.teach_train.torch()
        self.teach_test = self.teach_test.torch()
        self.teach_valid = self.teach_valid.torch()
        self.student_data = self.student_data.torch()
        self.student_test = self.student_test.torch()

    def get_loader(self, subset: str, shuffle: bool = None) -> DataLoader:
        if subset == "test":
            inputs = self.student_test.data[self.test_inds]
            labels = self.student_test.targets[self.test_inds]
        else:
            subset_inds = self.main_inds[subset]
            inputs = self.student_data.data[subset_inds]
            labels = self.student_data.targets[subset_inds]

        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if self.batch_sizes[subset] == -1:
            batch_size = len(inputs)
        else:
            batch_size = self.batch_sizes[subset]

        if shuffle is None:
            shuffle = subset in {"train", "target"}

        loader = DataLoader(
            dataset=TensorDataset(inputs, labels),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=(subset == "train"),
            **self.loader_kwargs,
        )

        return loader

    def move_from_pool_to_train(self, pool_inds_to_move: Union[int, Sequence[int]]) -> None:
        """
        Important:
        - pool_inds_to_move and pool_inds_to_keep index into self.main_inds["pool"]
        - self.main_inds["pool"] and train_inds_to_add index into self.main_dataset
        """

        # We can add the labelling here or in main (probably easier to do in main)
        # so we can keep as much of their code the same.

        if isinstance(pool_inds_to_move, int):
            pool_inds_to_move = [pool_inds_to_move]

        pool_inds_to_keep = range(len(self.main_inds["pool"]))
        pool_inds_to_keep = np.setdiff1d(pool_inds_to_keep, pool_inds_to_move)

        train_inds_to_add = [self.main_inds["pool"][ind] for ind in pool_inds_to_move]

        self.main_inds["pool"] = [self.main_inds["pool"][ind] for ind in pool_inds_to_keep]
        self.main_inds["train"] += train_inds_to_add

    def overwrite_labels_by_indices(self, indices, aggregator, votes):
        """
        Takes a sequence of indices (of the original dataset) and re-labels
        them in a private way, using some aggregation method.

        Important:
        - indices indexes into self.main_inds["pool"]
        - self.main_inds["pool"] indexes into self.main_dataset
        """

        dat_dep_costs = []
        for i in indices:
            # get the index of the element in the main dataset (self.student_data)
            main_index = self.main_inds["pool"][i]
            dat_dep_costs.append(data_dependent_cost(votes[main_index], aggregator.num_labels, aggregator.scale))
            self.student_data.targets[main_index] = aggregator.aggregate(votes[main_index])

        return dat_dep_costs



def initialize_indices(
    labels: np.ndarray, label_counts: Union[ConfigDict, ConfigList, int], rng: Generator
) -> np.ndarray:
    
    # We want to change this from going based on labels to going based on sizes of data because
    # going based on labels is very difficult privately
    if isinstance(label_counts, int):
        if label_counts == -1:
            label_counts = len(labels)

        selected_inds = rng.choice(len(labels), size=label_counts, replace=False)

    else:
        if isinstance(label_counts, (list, ListConfig)):
            label_counts = preprocess_label_counts(label_counts, rng)

        selected_inds = []

        for _class, count in label_counts.items():
            class_inds = np.flatnonzero(labels == _class)

            if count == -1:
                count = len(class_inds)

            selected_inds += [rng.choice(class_inds, size=count, replace=False)]

        selected_inds = np.concatenate(selected_inds)
        selected_inds = rng.permutation(selected_inds)

    return selected_inds


def preprocess_label_counts(label_counts: ConfigList, rng: Generator) -> dict:
    processed_label_counts = {}

    for cfg in label_counts:
        classes = eval(cfg["classes"])
        n_classes = cfg["n_classes"]

        if (n_classes != -1) and (n_classes < len(classes)):
            classes = rng.choice(list(classes), size=n_classes, replace=False)

        for _class in classes:
            assert _class not in processed_label_counts
            processed_label_counts[_class] = cfg["n_per_class"]

    return processed_label_counts


def map_classes(dataset: Dataset, class_map: Union[ConfigDict, ConfigList]) -> Dataset:
    class_map = preprocess_class_map(class_map)

    dataset.original_targets = dataset.targets
    dataset.targets = class_map(dataset.targets)

    return dataset


def preprocess_class_map(class_map: Union[ConfigDict, ConfigList]) -> Callable:
    if isinstance(class_map, (list, ListConfig)):
        processed_class_map = {}

        for cfg in class_map:
            old_class = eval(cfg["old_class"])

            if isinstance(old_class, int):
                old_class = [old_class]

            for _class in old_class:
                assert _class not in processed_class_map
                processed_class_map[_class] = cfg["new_class"]

    else:
        processed_class_map = dict(class_map)

    return np.vectorize(processed_class_map.get)
